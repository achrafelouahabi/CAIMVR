import argparse
import itertools
import time
import torch

from model import CAIMVR
from utils.get_mask import get_mask
from utils.util import cal_std
from utils.logger_ import get_logger
from utils.datasets_clustering import *
from configure.configure_clustering import get_default_config
import collections
import warnings
import json

warnings.simplefilter("ignore")

dataset = {
    0: "Caltech101-20",
    1: "Scene_15",
    2: "NoisyMNIST",
    3: "LandUse_21",
    4: "MSRC_v1",
    5: "hand"
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print losses')
parser.add_argument('--eval_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['eval_num'] = args.eval_num
    config['dataset'] = dataset
    logger, plt_name = get_logger(config)

    logger.info('Dataset:' + str(dataset))
    np.random.seed(1)
    random.seed(1)
    # Load data
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]


    print(np.shape(x1_train_raw))
    print(np.shape(x2_train_raw))
    # print(np.shape(x3_train_raw))

    HISTORIQUES_FOLDER = "historiques"
    RESULTS_FOLDER = "results"
    os.makedirs(HISTORIQUES_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)


    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))
    
    fold_acc, fold_nmi, fold_ari = [], [], []
      


    for data_seed in range(1, args.test_time + 1):
        
        start = time.time()
        np.random.seed(data_seed)

        # Get Mask
        mask = get_mask(2, x1_train_raw.shape[0], config['missing_rate'])

        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        mask = torch.from_numpy(mask).long().to(device)

        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        if config['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['seed']

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build model
        CAIMVR_model = CAIMVR(config)
        optimizer = torch.optim.Adam(
            itertools.chain(
              CAIMVR_model.autoencoder1.parameters(),
              CAIMVR_model.autoencoder2.parameters(),
              CAIMVR_model.img2txt.parameters(), 
              CAIMVR_model.txt2img.parameters()),
            lr=config['training']['lr'])

        # Print the models
        logger.info(CAIMVR_model.autoencoder1)
        logger.info(CAIMVR_model.img2txt)
        logger.info(optimizer)

        CAIMVR_model.autoencoder1.to(device), CAIMVR_model.autoencoder2.to(device)
        CAIMVR_model.img2txt.to(device), CAIMVR_model.txt2img.to(device)
        # Avant l'appel à train_clustering
        # if data_seed > 1:
        #     ckpt = torch.load("weights_seed1.pth", map_location=device)
        #     CAIMVR_model.autoencoder1.load_state_dict(ckpt['autoencoder1'])
        #     CAIMVR_model.autoencoder2.load_state_dict(ckpt['autoencoder2'])
        #     CAIMVR_model.img2txt.load_state_dict(ckpt['img2txt'])
        #     CAIMVR_model.txt2img.load_state_dict(ckpt['txt2img'])
        #     optimizer.load_state_dict(ckpt['optimizer'])

            
            # print(f">>> Poids seed1 rechargés pour le seed {data_seed}")

        # Training
        acc, nmi, ari ,results ,epoch_pos = CAIMVR_model.train_clustering(config, logger, accumulated_metrics, x1_train, x2_train, Y_list, mask,
                                          optimizer, device ,data_seed)
        if data_seed == 1:
            torch.save({
                'autoencoder1': CAIMVR_model.autoencoder1.state_dict(),
                'autoencoder2': CAIMVR_model.autoencoder2.state_dict(),
                'img2txt':      CAIMVR_model.img2txt.state_dict(),
                'txt2img':      CAIMVR_model.txt2img.state_dict(),
                'optimizer':    optimizer.state_dict()
            }, "weights_seed1.pth")
            print(">>> Poids du seed 1 sauvegardés dans weights_seed1.pth")



        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)

        
        # Calculez la durée écoulée en secondes
        duration_seconds = time.time() - start

        # Convertissez la durée en heures, minutes et secondes
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Affichez la durée formatée en heures:minutes:secondes
        print(f"la durée de test N° {int(data_seed)}: {int(hours)}:{int(minutes)}:{int(seconds)}")
  

    logger.info('--------------------Training over--------------------')
    acc, nmi, ari = cal_std(fold_acc, fold_nmi, fold_ari)                                                 


if __name__ == '__main__':
    main()
