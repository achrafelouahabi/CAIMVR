from __future__ import print_function, absolute_import, division
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import instance_distribution_alignment, supervised_discriminative
from utils import clustering, classify
from utils.next_batch import next_batch_gt, next_batch
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# --------------------------------------------------------------------
# Helpers
def exists(x): return x is not None
def default(val, d): return val if exists(val) else d

# --------------------------------------------------------------------
# Multi-head attention (restored from original model.py)
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        x_reshaped = x.unsqueeze(1) # (b, 1, d)
        context_reshaped = context.unsqueeze(1) if context is not None else x_reshaped # (b, 1, d)

        if kv_include_self:
            context_reshaped = torch.cat((x_reshaped, context_reshaped), dim=1)

        q = self.to_q(x_reshaped) # (b, 1, inner_dim)
        k, v = self.to_kv(context_reshaped).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out).squeeze(1) # (b, dim)

# --------------------------------------------------------------------
# Attention module with residual connection
class EncodeView(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = Attention(input_dim, heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Projection for residual if dimensions don't match
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, view1):
        # Residual connection
        identity=view1
        identity = self.projection(identity)
        identity=self.norm(identity)
        
        # Attention
        attn_output = self.attention(view1, context=view1, kv_include_self=True)
        attn_output=self.norm(attn_output)
        attn_output=self.act(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Combine with residual
        combined = identity + attn_output
        
        # Post-processing
        enco_attn = self.linear(combined)
        enco_attn = self.norm(enco_attn)
        enco_attn = self.act(enco_attn)
        
        return enco_attn

# --------------------------------------------------------------------
# Autoencoder without cross-attention
class Autoencoder(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True,
                 heads=8):
        super().__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm
        self.heads = heads

        # Encoder
        enc_layers = []
        for i in range(self._dim):
            enc_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if batchnorm: enc_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                enc_layers.append(self._act())
        self._encoder = nn.Sequential(*enc_layers)
        self.attention=EncodeView(encoder_dim[-1],encoder_dim[-1],self.heads)
        self.soft=nn.Softmax(dim=1)
        # Decoder
        dec_layers = []
        rev = list(reversed(encoder_dim))
        for i in range(self._dim):
            dec_layers.append(nn.Linear(rev[i], rev[i + 1]))
            if batchnorm: dec_layers.append(nn.BatchNorm1d(rev[i + 1]))
            dec_layers.append(self._act())
        self._decoder = nn.Sequential(*dec_layers)

    def _act(self):
        if self._activation == 'relu': return nn.ReLU()
        if self._activation == 'Prelu': return nn.PReLU()
        if self._activation == 'leakyrelu': return nn.LeakyReLU(0.2, inplace=True)
        if self._activation == 'tanh': return nn.Tanh()
        if self._activation == 'gelu': return nn.GELU()
        raise ValueError(f"Unknown activation: {self._activation}")

    def encoder(self, x):
        latent_x = self._encoder(x)
        attn_x=self.attention(latent_x)
        return self.soft(latent_x),attn_x

    def decoder(self, z): return self._decoder(z)

    def forward(self, x):
        z, xz = self.encoder(x)
        return z, self.decoder(xz)

# --------------------------------------------------------------------
# Cross-view predictor
class ViewPrediction(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = Attention(input_dim, heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, view1, view2):
        identity1 = view1
        attn_output = self.attention(view1, context=view2, kv_include_self=True)
        attn_output = self.dropout(attn_output)
        attn_output = identity1 + attn_output
        attn_output = self.norm1(attn_output)
        attn_output = self.act(attn_output)
        
        identity2 = self.projection(attn_output)
        predicted_view = self.linear(attn_output)
        predicted_view = identity2 + predicted_view
        predicted_view = self.norm2(predicted_view)
        predicted_view = self.act(predicted_view)
        
        return predicted_view

# --------------------------------------------------------------------
# Prediction module
class Prediction(nn.Module):
    def __init__(self, prediction_dim, activation='gelu',
                 batchnorm=True, heads=8):
        super().__init__()
        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        enc_layers = []
        for i in range(self._depth):
            enc_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                enc_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if i < self._depth - 1:
                enc_layers.append(self._get_activation_layer())
        
        self._encoder = nn.Sequential(*enc_layers)
        
        self.cross = ViewPrediction(self._prediction_dim[-1], self._prediction_dim[-1], heads)

        dec_layers = []
        for i in range(self._depth, 0, -1):
            dec_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if batchnorm:
                dec_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
            if i > 0:
                dec_layers.append(self._get_activation_layer())
        dec_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*dec_layers)

    def _get_activation_layer(self):
        if self._activation == 'relu': return nn.ReLU()
        if self._activation == 'leakyrelu': return nn.LeakyReLU(0.2, inplace=True)
        if self._activation == 'tanh': return nn.Tanh()
        if self._activation == 'prelu': return nn.PReLU()
        if self._activation == 'gelu': return nn.GELU()
        return nn.ReLU()

    def forward(self, x, y=None):
        latent_x = self._encoder(x)
        latent_y = self._encoder(y) if y is not None else latent_x
        latent = self.cross(latent_x, latent_y)
        output = self._decoder(latent)
        return output, latent_y

# --------------------------------------------------------------------
class CAIMVR:
    """
    Dual Contrastive Prediction with graceful handling of missing views.
    """
    def __init__(self, config):
        self._config = config

        if config['Autoencoder']['arch1'][-1] != config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dimension between the two autoencoders.')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + config['Prediction']['arch2']

        self.autoencoder1 = Autoencoder(
            config['Autoencoder']['arch1'],
            activation=config['Autoencoder']['activations1'],
            batchnorm=config['Autoencoder']['batchnorm'],
            heads=config['Autoencoder']['heads'],
        )
        self.autoencoder2 = Autoencoder(
            config['Autoencoder']['arch2'],
            activation=config['Autoencoder']['activations2'],
            batchnorm=config['Autoencoder']['batchnorm'],
            heads=config['Autoencoder']['heads'],
        )

        self.img2txt = Prediction(self._dims_view1, heads=config['Prediction']['heads'],activation=config['Prediction']['activations1'])
        self.txt2img = Prediction(self._dims_view2, heads=config['Prediction']['heads'],activation=config['Prediction']['activations2'])

    # ========================================================================
    # Fonctions utilitaires pour geler/dégeler
    # ========================================================================
    def _freeze_module(self, module):
        """Geler tous les paramètres d'un module."""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze_module(self, module):
        """Dégeler tous les paramètres d'un module."""
        for param in module.parameters():
            param.requires_grad = True
        module.train()

    # ------------------------------------------------------------------
    def train_clustering(
        self, config, logger, metrics, x1_train, x2_train,
        Y_list, mask, optimizer, device, test_id
    ):
        """
        Training en TROIS phases :
        - Phase 1 : reconstruction seule
        - Phase 2 : ajout dual prediction + contrastive
        - Phase 3 : focus sur missing views (autoencoders gelés)
        """
        total_epochs = config['training']['epoch']
        pretrain_epochs = config['training']['pretrain_epochs']
        phase3_epochs = config['training'].get('epoch_3', 0)  # Par défaut 0 si non défini
        finetune_epochs = total_epochs
        batch_size = config['training']['batch_size']

        # Init metrics
        for key in ['acc', 'nmi', 'ARI', 'f_measure', 'all1', 'all2', 'map1', 'map2', 'all_icl']:
            if key not in metrics:
                metrics[key] = []

        # Samples avec 2 vues
        full_mask = (mask == torch.LongTensor([1, 1]).to(device)).all(dim=1)
        train_view1, train_view2 = x1_train[full_mask], x2_train[full_mask]
        n_samples = train_view1.size(0)

        # -------------------------------
        # Phase 1 : Pré-entrainement
        # -------------------------------
        print("\n=== Phase 1: Pretraining (Reconstruction only) ===")
        for epoch in range(pretrain_epochs):
            shuffled_indices = torch.randperm(train_view1.size(0))
            v1, v2 = train_view1[shuffled_indices], train_view2[shuffled_indices]

            loss_total = loss_r1 = loss_r2 = 0

            for bx1, bx2, _ in next_batch(v1, v2, batch_size):
                bx1, bx2 = bx1.to(device), bx2.to(device)

                # Encoders
                z1, xz1 = self.autoencoder1.encoder(bx1)
                z2, xz2 = self.autoencoder2.encoder(bx2)

                # Reconstruction
                x_hat1 = self.autoencoder1.decoder(xz1)
                x_hat2 = self.autoencoder2.decoder(xz2)
                mp_loss1 = F.mse_loss(x_hat1, bx1)
                mp_loss2 = F.mse_loss(x_hat2, bx2)

                # Total loss = reconstruction
                all_loss = config['training']['lambda2'] * (mp_loss1 + mp_loss2)

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_total += all_loss.item()
                loss_r1 += mp_loss1.item()
                loss_r2 += mp_loss2.item()

            if (epoch + 1) % config['print_num'] == 0:
                print(
                    f"[Pretrain] Test {test_id} - Epoch {epoch+1}/{pretrain_epochs} | "
                    f"mp_loss1={loss_r1:.4f} mp_loss2={loss_r2:.4f} Total={loss_total:.4e}"
                )

        # -------------------------------
        # Phase 2 : Fine-tuning complet
        # -------------------------------
        print("\n=== Phase 2: Fine-tuning (Full training) ===")
        for epoch in range(finetune_epochs):
            shuffled_indices = torch.randperm(train_view1.size(0))
            v1, v2 = train_view1[shuffled_indices], train_view2[shuffled_indices]

            loss_total = loss_r1 = loss_r2 = loss_map1 = loss_map2 = loss_ida_total = 0

            for bx1, bx2, _ in next_batch(v1, v2, batch_size):
                bx1, bx2 = bx1.to(device), bx2.to(device)

                # Encoders
                z1, xz1 = self.autoencoder1.encoder(bx1)
                z2, xz2 = self.autoencoder2.encoder(bx2)

                # Reconstructions
                x_hat1 = self.autoencoder1.decoder(xz1)
                x_hat2 = self.autoencoder2.decoder(xz2)
                mp_loss1 = F.mse_loss(x_hat1, bx1)
                mp_loss2 = F.mse_loss(x_hat2, bx2)

                # instance distribution alignment
                loss_ida = instance_distribution_alignment(z1, z2, config['training']['alpha'])

                # Dual prediction
                pred12, z12 = self.img2txt(z1, z2)
                pred21, z21 = self.txt2img(z2, z1)

                z12 = self.autoencoder1.soft(z12)
                z21 = self.autoencoder1.soft(z21)

                bp_loss = F.mse_loss(pred12, z2) + F.mse_loss(pred21, z1)
                loss_lsc = instance_distribution_alignment(z12, z21, 1)

                # Total loss
                all_loss = (
                    loss_ida +
                    config['training']['lambda2'] * (mp_loss1 + mp_loss2) +
                    config['training']['lambda1'] * bp_loss +
                    loss_lsc
                )

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                # Logs
                loss_total += all_loss.item()
                loss_r1 += mp_loss1.item()
                loss_r2 += mp_loss2.item()
                loss_map1 += F.mse_loss(pred12, z2).item()
                loss_map2 += F.mse_loss(pred21, z1).item()
                loss_ida_total += loss_ida.item()

            if (epoch + 1) % config['print_num'] == 0:
                print(
                    f"[Finetune] Test {test_id} - Epoch {epoch+1}/{finetune_epochs} | "
                    f"mp_loss1={loss_r1:.4f} mp_loss2={loss_r2:.4f} "
                    f"Map1={loss_map1:.4f} Map2={loss_map2:.4f} "
                    f"ICL={loss_ida_total:.4e} Total={loss_total:.4e}"
                )

            # Eval pendant fine-tuning
            if (epoch + 1) % config['eval_num'] == 0:
                self.evaluate(
                    x1_train, x2_train, Y_list, mask, device,
                    metrics, logger, loss_r1, loss_r2,
                    loss_map1, loss_map2, loss_ida_total
                )

        # -------------------------------
        # Phase 3 : Missing views
        # -------------------------------
        if phase3_epochs > 0 and config.get('missing_rate', 0) != 0:
            print(f"\n=== Phase 3: Training predictors with missing views ===")
            print(">>> FREEZING: autoencoder1 and autoencoder2")
            print(">>> UNFREEZING: img2txt and txt2img")
            
            # Geler les autoencoders
            self._freeze_module(self.autoencoder1)
            self._freeze_module(self.autoencoder2)
            
            # Dégeler les prédicteurs
            self._unfreeze_module(self.img2txt)
            self._unfreeze_module(self.txt2img)
            
            # Optimiseur seulement pour prédicteurs
            pred_params = list(self.img2txt.parameters()) + list(self.txt2img.parameters())
            optimizer_pred = torch.optim.Adam(pred_params, lr=config['training']['lr']*10)
            
            for epoch in range(phase3_epochs):
                shuffled_indices = torch.randperm(n_samples)
                v1, v2 = train_view1[shuffled_indices], train_view2[shuffled_indices]
                
                loss_total = loss_v1_only = loss_v2_only = 0
                n_v1_only = n_v2_only = 0
                
                for bx1, bx2, _ in next_batch(v1, v2, batch_size):
                    bx1, bx2 = bx1.to(device), bx2.to(device)
                    batch_len = bx1.size(0)
                    
                    # Alternance symétrique
                    rand_mask = torch.rand(batch_len, device=device) > 0.5
                    view1_mask = rand_mask
                    view2_mask = ~rand_mask
                    
                    batch_loss = 0
                    
                    # Cas 1 : Vue1 présente, Vue2 manquante
                    v1_only = view1_mask
                    if v1_only.any():
                        n_v1_only += v1_only.sum().item()
                        
                        with torch.no_grad():
                            z1, _ = self.autoencoder1.encoder(bx1[v1_only])
                            z2_true, _ = self.autoencoder2.encoder(bx2[v1_only])
                        
                        pred_z2, _ = self.img2txt(z1, z1)
                        loss = F.mse_loss(pred_z2, z2_true)
                        batch_loss += loss
                        loss_v1_only += loss.item()
                    
                    # Cas 2 : Vue2 présente, Vue1 manquante
                    v2_only = ~view1_mask
                    if v2_only.any():
                        n_v2_only += v2_only.sum().item()
                        
                        with torch.no_grad():
                            z2, _ = self.autoencoder2.encoder(bx2[v2_only])
                            z1_true, _ = self.autoencoder1.encoder(bx1[v2_only])
                        
                        pred_z1, _ = self.txt2img(z2, z2)
                        loss = F.mse_loss(pred_z1, z1_true)
                        batch_loss += loss
                        loss_v2_only += loss.item()
                    
                    # Backprop uniquement sur les prédicteurs
                    if batch_loss > 0:
                        optimizer_pred.zero_grad()
                        batch_loss.backward()
                        optimizer_pred.step()
                        loss_total += batch_loss.item()
                
                if (epoch + 1) % config['print_num'] == 0:
                    print(
                        f"[Phase 3] Epoch {pretrain_epochs+finetune_epochs+epoch+1} | "
                        f"V1_only={loss_v1_only:.4f} V2_only={loss_v2_only:.4f} | "
                        f"Samples: V1={n_v1_only} V2={n_v2_only}"
                    )
                
                # Évaluation
                if (epoch + 1) % config['eval_num'] == 0:
                    self.evaluate(
                        x1_train, x2_train, Y_list, mask, device,
                        metrics, logger, 0, 0, loss_v1_only, loss_v2_only, 0
                    )
        
        idx_best = metrics['acc'].index(max(metrics['acc'])) if metrics['acc'] else 0
        return (
            metrics['acc'][idx_best] if metrics['acc'] else 0,
            metrics['nmi'][idx_best] if metrics['nmi'] else 0,
            metrics['ARI'][idx_best] if metrics['ARI'] else 0,
            metrics,
            (idx_best + 1) * config['eval_num'] if metrics['acc'] else 0,
        )

    # ------------------------------------------------------------------
    def evaluate(self, x1, x2, Y, mask, device, metrics,
                 logger, r1, r2, m1, m2, icl):
        """Evaluation that supports missing-view fallback."""
        self.autoencoder1.eval()
        self.autoencoder2.eval()
        self.img2txt.eval()
        self.txt2img.eval()

        img_idx = (mask[:, 0] == 1).nonzero(as_tuple=True)[0]
        txt_idx = (mask[:, 1] == 1).nonzero(as_tuple=True)[0]
        img_miss_idx = (mask[:, 0] == 0).nonzero(as_tuple=True)[0]
        txt_miss_idx = (mask[:, 1] == 0).nonzero(as_tuple=True)[0]

        z_img_val = torch.tensor([], device=device)
        xz_img_val = torch.tensor([], device=device)
        if img_idx.numel() > 0:
            z_img_val, xz_img_val = self.autoencoder1.encoder(x1[img_idx].to(device))
        
        z_txt_val = torch.tensor([], device=device)
        xz_txt_val = torch.tensor([], device=device)
        if txt_idx.numel() > 0:
            z_txt_val, xz_txt_val = self.autoencoder2.encoder(x2[txt_idx].to(device))

        latent_img = torch.zeros(x1.size(0), self._latent_dim, device=device)
        latent_txt = torch.zeros(x2.size(0), self._latent_dim, device=device)

        with torch.no_grad():
            if img_miss_idx.numel() > 0:
                txt_for_img, _ = self.autoencoder2.encoder(x2[img_miss_idx].to(device))
                pred_img, _ = self.txt2img(txt_for_img)
                latent_img[img_miss_idx] = pred_img
            if txt_miss_idx.numel() > 0:
                img_for_txt, _ = self.autoencoder1.encoder(x1[txt_miss_idx].to(device))
                pred_txt, _ = self.img2txt(img_for_txt)
                latent_txt[txt_miss_idx] = pred_txt

        latent_img[img_idx] = z_img_val
        latent_txt[txt_idx] = z_txt_val

        fused = torch.cat([latent_img, latent_txt], dim=1).detach().cpu().numpy()
        scores = clustering.get_score(
            [fused], Y, metrics['acc'], metrics['nmi'],
            metrics['ARI'], metrics['f_measure']
        )

        metrics['all1'].append(r1)
        metrics['all2'].append(r2)
        metrics['map1'].append(m1)
        metrics['map2'].append(m2)
        metrics['all_icl'].append(icl)

        print("#trainingset_view_concat " + str(scores))

        self.autoencoder1.train()
        self.autoencoder2.train()
        self.img2txt.train()
        self.txt2img.train()
        
    def train_supervised(self, config, logger, accumulated_metrics, x1_train, x2_train, 
                        x1_test, x2_test, labels_train, labels_test, mask_train, 
                        mask_test, optimizer, device,test_id):
        """
        Two-phase supervised training:
        - Phase 1 (pretrain_epochs): reconstruction + instance distribution alignment
        - Phase 2 (finetune_epochs): adds dual prediction + supervised discriminative
        """
        total_epochs = config['training']['epoch']
        pretrain_epochs = config['training']['pretrain_epochs']  
        finetune_epochs = total_epochs  
        batch_size = config['training']['batch_size']
        
        # Initialize metrics
        for key in ['acc', 'precision', 'f_measure']:
            if key not in accumulated_metrics:
                accumulated_metrics[key] = []
        
        # Get complete data for training
        full_mask = (mask_train == torch.LongTensor([1, 1]).to(device)).all(dim=1)
        train_view1 = x1_train[full_mask]
        train_view2 = x2_train[full_mask]
        GT = torch.from_numpy(labels_train).long().to(device)[full_mask]
        
        classes = np.unique(np.concatenate([labels_train, labels_test])).size
        flag_gt = (torch.min(GT) == 1)
        
        # -------------------------------
        # Phase 1: Pretraining
        # -------------------------------
        print("\n=== Starting Phase 1: Pretraining ===")
        for epoch in range(pretrain_epochs):
            X1, X2, gt = shuffle(train_view1, train_view2, GT)
            
            loss_total = loss_r1 = loss_r2 = loss_ida_total = 0
            
            for batch_x1, batch_x2, gt_batch, _ in next_batch_gt(X1, X2, gt, batch_size):
                batch_x1, batch_x2 = batch_x1.to(device), batch_x2.to(device)
                
                # Encoders
                z1, xz1 = self.autoencoder1.encoder(batch_x1)
                z2, xz2 = self.autoencoder2.encoder(batch_x2)
                
                # Reconstruction
                x_hat1 = self.autoencoder1.decoder(xz1)
                x_hat2 = self.autoencoder2.decoder(xz2)
                mp_loss1 = F.mse_loss(x_hat1, batch_x1)
                mp_loss2 = F.mse_loss(x_hat2, batch_x2)
                
                # instance distribution alignment
                loss_ida = instance_distribution_alignment(z1, z2, config['training']['alpha'])
                
                # Total loss = reconstruction + instance distribution alignment
                all_loss = (
                     
                    config['training']['lambda2'] * (mp_loss1 + mp_loss2)
                )
                
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                
                loss_total += all_loss.item()
                loss_r1 += mp_loss1.item()
                loss_r2 += mp_loss2.item()
                loss_ida_total += loss_ida.item()
            
            if (epoch + 1) % config['print_num'] == 0:
                output = (f"[Pretrain] Test {test_id} Epoch {epoch+1}/{pretrain_epochs} | "
                        f"mp_loss1={loss_r1:.4f} mp_loss2={loss_r2:.4f} "
                        f"ICL={loss_ida_total:.4e} Total={loss_total:.4e}")
                print("\033[2;29m" + output + "\033[0m")
                
                # Evaluation during pretraining
                self._evaluate_supervised(
                    x1_train, x2_train, x1_test, x2_test,
                    labels_train, labels_test, mask_train, mask_test,
                    config, device, accumulated_metrics, logger
                )
        
        # -------------------------------
        # Phase 2: Fine-tuning
        # -------------------------------
        print("\n=== Starting Phase 2: Fine-tuning ===")
        for epoch in range(finetune_epochs):
            X1, X2, gt = shuffle(train_view1, train_view2, GT)
            
            loss_total = loss_r1 = loss_r2 = 0
            loss_ida_total = loss_sd_total = loss_map1 = loss_map2 = 0
            
            for batch_x1, batch_x2, gt_batch, _ in next_batch_gt(X1, X2, gt, batch_size):
                batch_x1, batch_x2 = batch_x1.to(device), batch_x2.to(device)
                
                # Encoders
                z1, xz1 = self.autoencoder1.encoder(batch_x1)
                z2, xz2 = self.autoencoder2.encoder(batch_x2)
                
                # Reconstruction
                x_hat1 = self.autoencoder1.decoder(xz1)
                x_hat2 = self.autoencoder2.decoder(xz2)
                mp_loss1 = F.mse_loss(x_hat1, batch_x1)
                mp_loss2 = F.mse_loss(x_hat2, batch_x2)
                
                # instance distribution alignment
                loss_ida = instance_distribution_alignment(z1, z2, config['training']['alpha'])
                
                # supervised discriminative
                loss_sd = supervised_discriminative(
                    torch.cat([z1, z2], dim=1), gt_batch, classes, flag_gt
                )
                
                # Dual prediction
                pred12, z12 = self.img2txt(z1, z2)
                pred21, z21 = self.txt2img(z2, z1)
                
                z12 = self.autoencoder1.soft(z12)
                z21 = self.autoencoder1.soft(z21)
                
                bp_loss = F.mse_loss(pred12, z2) + F.mse_loss(pred21, z1)
                loss_lsc = instance_distribution_alignment(z12, z21, 1)
                
                # Total loss
                all_loss = (
                    loss_ida +
                    config['training']['lambda2'] * (mp_loss1 + mp_loss2) +
                    config['training']['lambda1'] * bp_loss +
                    loss_sd +
                    loss_lsc
                )
                
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                loss_total += all_loss.item()
                loss_r1 += mp_loss1.item()
                loss_r2 += mp_loss2.item()
                loss_map1 += F.mse_loss(pred12, z2).item()
                loss_map2 += F.mse_loss(pred21, z1).item()
                loss_ida_total += loss_ida.item()
                loss_sd_total += loss_sd.item()
            
            if (epoch + 1) % config['print_num'] == 0:
                output = (f"[Finetune] Test {test_id} Epoch {epoch+1}/{finetune_epochs} | "
                        f"mp_loss1={loss_r1:.4f} mp_loss2={loss_r2:.4f} "
                        f"Map1={loss_map1:.4f} Map2={loss_map2:.4f} "
                        f"ICL={loss_ida_total:.4e} sd={loss_sd_total:.4e} "
                        f"Total={loss_total:.4e}")
                print("\033[2;29m" + output + "\033[0m")
                
                # Evaluation during fine-tuning
                self._evaluate_supervised(
                    x1_train, x2_train, x1_test, x2_test,
                    labels_train, labels_test, mask_train, mask_test,
                    config, device, accumulated_metrics, logger
                )
        
        # Return best results

        idx_best = accumulated_metrics['acc'].index(max(accumulated_metrics['acc'])) if accumulated_metrics['acc'] else 0
        return (
            accumulated_metrics['acc'][idx_best] if accumulated_metrics['acc'] else 0,
            accumulated_metrics['precision'][idx_best] if accumulated_metrics['precision'] else 0,
            accumulated_metrics['f_measure'][idx_best] if accumulated_metrics['f_measure'] else 0,
            accumulated_metrics['auc'][idx_best] if accumulated_metrics['auc'] else 0,
            accumulated_metrics,
            (idx_best + 1) * config['eval_num'] if accumulated_metrics['acc'] else 0,
        )



    def _evaluate_supervised(self, x1_train, x2_train, x1_test, x2_test,
                            labels_train, labels_test, mask_train, mask_test,
                            config, device, accumulated_metrics, logger):
        """Helper function for evaluation during training."""
        with torch.no_grad():
            self.autoencoder1.eval()
            self.autoencoder2.eval()
            self.img2txt.eval()
            self.txt2img.eval()
            
            # Training data
            latent_fusion_train = self._encode_with_missing_views(
                x1_train, x2_train, mask_train, config, device
            )
            
            # Test data
            latent_fusion_test = self._encode_with_missing_views(
                x1_test, x2_test, mask_test, config, device
            )
            

            from sklearn.metrics import (accuracy_score, precision_score, 
                                        f1_score, roc_auc_score)


            label_pre, proba = classify.ave_with_proba(
                latent_fusion_train, 
                latent_fusion_test, 
                labels_train,
                n_neighbors=config['k']
            )


            acc = accuracy_score(labels_test, label_pre)
            precision = precision_score(labels_test, label_pre, average='macro')
            f_score = f1_score(labels_test, label_pre, average='macro')

            try:
                n_classes = len(np.unique(labels_train))
                
                if n_classes == 2:
                    classes = np.unique(labels_train)
                    pos_class_idx = np.where(classes == classes.max())[0][0]
                    auc = roc_auc_score(labels_test, proba[:, pos_class_idx])
                else:

                    auc = roc_auc_score(labels_test, proba, 
                                      multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Warning: Could not compute AUC - {e}")
                auc = 0.0
            
            # Accumuler les métriques
            accumulated_metrics['acc'].append(acc)
            accumulated_metrics['precision'].append(np.round(precision, 2))
            accumulated_metrics['f_measure'].append(np.round(f_score, 2))
            accumulated_metrics['auc'].append(np.round(auc, 2))
            
            print(f'\033[2;29m Accuracy: {acc:.4f} | '
                  f'Precision: {precision:.4f} | '
                  f'F-score: {f_score:.4f} | '
                  f'AUC: {auc:.4f}')
            
            self.autoencoder1.train()
            self.autoencoder2.train()
            self.img2txt.train()
            self.txt2img.train()

    def _encode_with_missing_views(self, x1, x2, mask, config, device):
        """Encode data handling missing views."""
        img_idx = (mask[:, 0] == 1).nonzero(as_tuple=True)[0]
        txt_idx = (mask[:, 1] == 1).nonzero(as_tuple=True)[0]
        img_miss_idx = (mask[:, 0] == 0).nonzero(as_tuple=True)[0]
        txt_miss_idx = (mask[:, 1] == 0).nonzero(as_tuple=True)[0]
        
        # Initialize latent codes
        latent_img = torch.zeros(x1.size(0), config['Autoencoder']['arch1'][-1]).to(device)
        latent_txt = torch.zeros(x2.size(0), config['Autoencoder']['arch2'][-1]).to(device)
        
        # Encode available views
        if img_idx.numel() > 0:
            z_img, _ = self.autoencoder1.encoder(x1[img_idx].to(device))
            latent_img[img_idx] = z_img
        
        if txt_idx.numel() > 0:
            z_txt, _ = self.autoencoder2.encoder(x2[txt_idx].to(device))
            latent_txt[txt_idx] = z_txt
        
        # Handle missing views via predictors
        if img_miss_idx.numel() > 0:
            z_txt_for_img, _ = self.autoencoder2.encoder(x2[img_miss_idx].to(device))
            pred_img, _ = self.txt2img(z_txt_for_img)
            latent_img[img_miss_idx] = pred_img
        
        if txt_miss_idx.numel() > 0:
            z_img_for_txt, _ = self.autoencoder1.encoder(x1[txt_miss_idx].to(device))
            pred_txt, _ = self.img2txt(z_img_for_txt)
            latent_txt[txt_miss_idx] = pred_txt
        
        # Fuse and return
        fused = torch.cat([latent_img, latent_txt], dim=1).cpu().numpy()
        return fused

    def train_HAR(self, config, logger, accumulated_metrics, train_data, optimizer, device):
        """
        Two-phase training for Human Action Recognition:
        - Phase 1 (pretrain_epochs): reconstruction + instance distribution alignment
        - Phase 2 (finetune_epochs): adds dual prediction + supervised discriminative
        
        Args:
            config: parameters defined in configure.py
            logger: print the information
            accumulated_metrics: list of metrics
            train_data: HAR data object
            optimizer: adam optimizer
            device: cuda device
        Returns:
            classification performance: RGB, Depth, RGB-D, onlyRGB, onlyDepth accuracy
        """
        total_epochs = config['training']['epoch']
        pretrain_epochs = config['training']['pretrain_epochs']  # e.g., 500
        finetune_epochs = total_epochs  # e.g., 1000
        batch_size = config['training']['batch_size']
        
        classes = train_data.cluster
        flag_gt = False
        
        # Initialize metrics
        for key in ['RGB', 'Depth', 'RGB-D', 'onlyRGB', 'onlyDepth']:
            if key not in accumulated_metrics:
                accumulated_metrics[key] = []
        
        # -------------------------------
        # Phase 1: Pretraining
        # -------------------------------
        print("\n=== Starting Phase 1: Pretraining (HAR) ===")
        for epoch in range(pretrain_epochs):
            loss_total = loss_r1 = loss_r2 = loss_ida_total = 0
            
            batch_x1, batch_x2, gt_batch = train_data.train_next_batch(batch_size)
            batch_x1 = torch.from_numpy(np.array(batch_x1)).float().to(device)
            batch_x2 = torch.from_numpy(np.array(batch_x2)).float().to(device)
            gt_batch = [np.argmax(one_hot) for one_hot in gt_batch]
            gt_batch = torch.from_numpy(np.array(gt_batch)).to(device)
            
            # Encoders
            z1, xz1 = self.autoencoder1.encoder(batch_x1)
            z2, xz2 = self.autoencoder2.encoder(batch_x2)
            
            # Reconstruction
            x_hat1 = self.autoencoder1.decoder(xz1)
            x_hat2 = self.autoencoder2.decoder(xz2)
            mp_loss1 = F.mse_loss(x_hat1, batch_x1)
            mp_loss2 = F.mse_loss(x_hat2, batch_x2)
            
            # instance distribution alignment
            loss_ida = instance_distribution_alignment(z1, z2, config['training']['alpha'])
            
            # Total loss = reconstruction + instance distribution alignment
            all_loss = (
                
                config['training']['lambda2'] * (mp_loss1 + mp_loss2)
            )
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            loss_total += all_loss.item()
            loss_r1 += mp_loss1.item()
            loss_r2 += mp_loss2.item()
            loss_ida_total += loss_ida.item()
            
            if (epoch + 1) % config['print_num'] == 0:
                output = (f"[Pretrain] Epoch {epoch+1}/{pretrain_epochs} | "
                        f"mp_loss1={loss_r1:.4f} mp_loss2={loss_r2:.4f} "
                        f"ICL={loss_ida_total:.4e} Total={loss_total:.4e}")
                print("\033[2;29m" + output + "\033[0m")
                
                # Evaluation during pretraining
                self._evaluate_har(train_data, config, device, accumulated_metrics, logger)
        
        # -------------------------------
        # Phase 2: Fine-tuning
        # -------------------------------
        print("\n=== Starting Phase 2: Fine-tuning (HAR) ===")
        for epoch in range(finetune_epochs):
            loss_total = loss_r1 = loss_r2 = 0
            loss_ida_total = loss_sd_total = loss_map1 = loss_map2 = 0
            
            batch_x1, batch_x2, gt_batch = train_data.train_next_batch(batch_size)
            batch_x1 = torch.from_numpy(np.array(batch_x1)).float().to(device)
            batch_x2 = torch.from_numpy(np.array(batch_x2)).float().to(device)
            gt_batch = [np.argmax(one_hot) for one_hot in gt_batch]
            gt_batch = torch.from_numpy(np.array(gt_batch)).to(device)
            
            # Encoders
            z1, xz1 = self.autoencoder1.encoder(batch_x1)
            z2, xz2 = self.autoencoder2.encoder(batch_x2)
            
            # Reconstruction
            x_hat1 = self.autoencoder1.decoder(xz1)
            x_hat2 = self.autoencoder2.decoder(xz2)
            mp_loss1 = F.mse_loss(x_hat1, batch_x1)
            mp_loss2 = F.mse_loss(x_hat2, batch_x2)
            
            # instance distribution alignment
            loss_ida = instance_distribution_alignment(z1, z2, config['training']['alpha'])
            
            # supervised discriminative
            loss_sd = supervised_discriminative(
                torch.cat([z1, z2], dim=1), gt_batch, classes, flag_gt
            )
            
            # Dual prediction with proper tensor reshaping
            
            pred12, z12 = self.img2txt(z1, z2)
            pred21, z21 = self.txt2img(z2, z1)

            z12 = self.autoencoder1.soft(z12)
            z21 = self.autoencoder1.soft(z21)
            
        
            loss_lsc = instance_distribution_alignment(z12, z21, 1)
            
            bp_loss = F.mse_loss(pred12, z2) + F.mse_loss(pred21, z1)
            
            # Total loss
            all_loss = (
                loss_ida +
                config['training']['lambda2'] * (mp_loss1 + mp_loss2) +
                config['training']['lambda1'] * bp_loss +
                loss_lsc+
                loss_sd
            )
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            loss_total += all_loss.item()
            loss_r1 += mp_loss1.item()
            loss_r2 += mp_loss2.item()
            loss_map1 += F.mse_loss(pred12, z2).item()
            loss_map2 += F.mse_loss(pred21, z1).item()
            loss_ida_total += loss_ida.item()
            loss_sd_total += loss_sd.item()
            
            if (epoch + 1) % config['print_num'] == 0:
                output = (f"[Finetune] Epoch {epoch+1}/{finetune_epochs} | "
                        f"mp_loss1={loss_r1:.4f} mp_loss2={loss_r2:.4f} "
                        f"Map1={loss_map1:.4f} Map2={loss_map2:.4f} "
                        f"ICL={loss_ida_total:.4e} sd={loss_sd_total:.4e} "
                        f"Total={loss_total:.4e}")
                print("\033[2;29m" + output + "\033[0m")
                
                # Evaluation during fine-tuning
                self._evaluate_har(train_data, config, device, accumulated_metrics, logger)
        

        idx_best = accumulated_metrics['RGB-D'].index(max(accumulated_metrics['RGB-D'])) if accumulated_metrics['RGB-D'] else 0
        return (
  
            accumulated_metrics['RGB'][idx_best] if accumulated_metrics['RGB'] else 0,
            accumulated_metrics['Depth'][idx_best] if accumulated_metrics['Depth'] else 0,
            accumulated_metrics['RGB-D'][idx_best] if accumulated_metrics['RGB-D'] else 0,
            accumulated_metrics['onlyRGB'][idx_best] if accumulated_metrics['onlyRGB'] else 0,
            accumulated_metrics['onlyDepth'][idx_best] if accumulated_metrics['onlyDepth'] else 0,
            accumulated_metrics,
            (idx_best + 1) * config['eval_num'] if accumulated_metrics['RGB'] else 0,
        )


    def _evaluate_har(self, train_data, config, device, accumulated_metrics, logger):
        """Helper function for HAR evaluation during training."""
        with torch.no_grad():
            self.autoencoder1.eval()
            self.autoencoder2.eval()
            self.img2txt.eval()
            self.txt2img.eval()
            
            # Prepare training data
            train_x1 = torch.from_numpy(np.array(train_data.train_data_y)).float().to(device)
            train_x2 = torch.from_numpy(np.array(train_data.train_data_x)).float().to(device)
            labels_train = [np.argmax(one_hot) for one_hot in train_data.train_data_label]
            labels_train = np.array(labels_train)
            
            # Encode training data
            z_train_1, _ = self.autoencoder1.encoder(train_x1)
            z_train_2, _ = self.autoencoder2.encoder(train_x2)
            
            latent_img_train = z_train_1.cpu().numpy()
            latent_txt_train = z_train_2.cpu().numpy()
            latent_fusion_train = torch.cat([z_train_1, z_train_2], dim=1).cpu().numpy()
            
            # Prepare test data
            test_x1 = torch.from_numpy(np.array(train_data.test_data_y)).float().to(device)
            test_x2 = torch.from_numpy(np.array(train_data.test_data_x)).float().to(device)
            labels_test = [np.argmax(one_hot) for one_hot in train_data.test_data_label]
            
            # Encode test data
            z_test_1, _ = self.autoencoder1.encoder(test_x1)
            z_test_2, _ = self.autoencoder2.encoder(test_x2)
            
            # RGB -> Depth reconstruction (R->D)
            
            pred_depth, _ = self.img2txt(z_test_1)

            latent_fusion_test_RD = torch.cat([z_test_1, pred_depth], dim=1).cpu().numpy()
            
            # Depth -> RGB reconstruction (D->R)
            pred_rgb, _ = self.txt2img(z_test_2)

            latent_fusion_test_DR = torch.cat([pred_rgb, z_test_2], dim=1).cpu().numpy()
            
            # Both views available (RGB+Depth)
            latent_fusion_test = torch.cat([z_test_1, z_test_2], dim=1).cpu().numpy()
            
            # Classification with sklearn
            from sklearn.metrics import accuracy_score
            
            # RGB only scenario (missing Depth)
            label_pre = classify.vote(latent_fusion_train, latent_fusion_test_RD, labels_train)
            scores_RD = accuracy_score(labels_test, label_pre)
            accumulated_metrics['RGB'].append(scores_RD)
            
            # Depth only scenario (missing RGB)
            label_pre = classify.vote(latent_fusion_train, latent_fusion_test_DR, labels_train)
            scores_DR = accuracy_score(labels_test, label_pre)
            accumulated_metrics['Depth'].append(scores_DR)
            
            # RGB+Depth (complete data)
            label_pre = classify.vote(latent_fusion_train, latent_fusion_test, labels_train)
            scores = accuracy_score(labels_test, label_pre)
            accumulated_metrics['RGB-D'].append(scores)
            
            # Only RGB (no fusion)
            label_pre = classify.vote(latent_img_train, z_test_1.cpu().numpy(), labels_train)
            scores_onlyrgb = accuracy_score(labels_test, label_pre)
            accumulated_metrics['onlyRGB'].append(scores_onlyrgb)
            
            # Only Depth (no fusion)
            label_pre = classify.vote(latent_txt_train, z_test_2.cpu().numpy(), labels_train)
            scores_onlydepth = accuracy_score(labels_test, label_pre)
            accumulated_metrics['onlyDepth'].append(scores_onlydepth)
            
            # Log results
            print(f'\033[2;29m RGB   Accuracy: {scores_RD:.4f}')
            print(f'\033[2;29m Depth Accuracy: {scores_DR:.4f}')
            print(f'\033[2;29m RGB+D Accuracy: {scores:.4f}')
            print(f'\033[2;29m onlyRGB Accuracy: {scores_onlyrgb:.4f}')
            print(f'\033[2;29m onlyDepth Accuracy: {scores_onlydepth:.4f}')
            
            self.autoencoder1.train()
            self.autoencoder2.train()
            self.img2txt.train()
            self.txt2img.train()

        




