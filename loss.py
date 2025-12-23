import sys
import torch

def compute_joint(x_out, x_tf_out):
    """
    Computes the symmetric and normalized joint distribution between two sets of outputs.
    
    Args:
        x_out: Tensor of shape [batch_size, k] - First outputs
        x_tf_out: Tensor of shape [batch_size, k] - Second outputs (transformed)
    
    Returns:
        p_i_j: Tensor of shape [k, k] - Normalized joint distribution
    """
    batch_size, k = x_out.size()
    assert x_tf_out.size(0) == batch_size and x_tf_out.size(1) == k, \
        "Dimensions of x_out and x_tf_out must match"
    
    # Compute joint distribution: outer product then sum over batch
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # [batch_size, k, k]
    p_i_j = p_i_j.sum(dim=0)  # [k, k]
    
    # Symmetrize the matrix
    p_i_j = (p_i_j + p_i_j.t()) / 2.0
    
    # Normalize to obtain a probability distribution
    p_i_j = p_i_j / p_i_j.sum()
    
    return p_i_j


def instance_distribution_alignment(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """
    Computes the contrastive loss to maximize consistency between two views.
    Based on mutual information with regularization of marginal distributions.
    
    Args:
        x_out: Tensor of shape [batch_size, k] - First outputs
        x_tf_out: Tensor of shape [batch_size, k] - Second outputs (transformed)
        lamb: float - Weighting coefficient for regularization
        EPS: float - Minimum value to avoid log(0)
    
    Returns:
        loss: Scalar tensor - Alignment loss
    """
    _, k = x_out.size()
    
    # Compute joint distribution
    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (k, k), "Joint distribution must be of size [k, k]"
    
    # Compute marginal distributions
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # Marginal over i
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # Marginal over j
    
    # Avoid null values for logarithm
    p_i_j = torch.clamp(p_i_j, min=EPS)
    p_j = torch.clamp(p_j, min=EPS)
    p_i = torch.clamp(p_i, min=EPS)
    
    # Compute loss: - sum(p_i_j * (log(p_i_j) - lamb*log(p_j) - lamb*log(p_i)))
    loss = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    loss = loss.sum()
    
    return loss


def supervised_discriminative(repre, gt, num_classes, flag_gt=False):
    """
    Computes the category-level contrastive loss (supervised).
    Encourages representations of the same class to be similar.
    
    Args:
        repre: Tensor of shape [N, D] - Representations/embeddings
        gt: Tensor of shape [N] or [N, 1] - Ground truth labels
        num_classes: int - Total number of classes
        flag_gt: bool - If True, subtracts 1 from labels (to go from 1-indexed to 0-indexed)
    
    Returns:
        loss: Scalar tensor - Discriminative loss
    """
    # Adjust labels if necessary
    if flag_gt:
        gt = gt - 1
    
    # Ensure gt is of shape [N]
    if gt.dim() > 1:
        gt = gt.squeeze()
    
    batch_size = gt.size(0)
    
    # Compute similarity matrix F_h_h = repre @ repre^T
    F_h_h = torch.matmul(repre, repre.t())  # [N, N]
    
    # Remove diagonal (self-similarity)
    F_hn_hn = torch.diag(F_h_h)
    F_h_h = F_h_h - torch.diag_embed(F_hn_hn)
    
    # One-hot encoding of labels
    label_onehot = torch.nn.functional.one_hot(gt, num_classes).float()  # [N, num_classes]
    
    # Number of samples per class
    label_num = torch.sum(label_onehot, dim=0, keepdim=True)  # [1, num_classes]
    
    # Sum of similarities for each class
    F_h_h_sum = torch.matmul(F_h_h, label_onehot)  # [N, num_classes]
    
    # Number of samples per class (excluding current sample)
    label_num_broadcast = label_num.repeat(batch_size, 1) - label_onehot  # [N, num_classes]
    label_num_broadcast = torch.clamp(label_num_broadcast, min=1)  # Avoid division by zero
    
    # Average similarity with each class
    F_h_h_mean = F_h_h_sum / label_num_broadcast  # [N, num_classes]
    
    # Prediction: class with maximum average similarity
    gt_pred = torch.argmax(F_h_h_mean, dim=1)  # [N]
    
    # Maximum average similarity (with any class)
    F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0]  # [N]
    
    # Correct prediction indicator
    theta = (gt == gt_pred).float()  # [N]
    
    # Average similarity with true class
    F_h_hn_mean = torch.sum(F_h_h_mean * label_onehot, dim=1)  # [N]
    
    # Loss: sum(relu(theta + (F_h_h_mean_max - F_h_hn_mean)))
    # Penalizes when the true class doesn't have the highest similarity
    margin = theta + (F_h_h_mean_max - F_h_hn_mean)
    loss = torch.sum(torch.relu(margin))
    
    return loss