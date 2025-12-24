import argparse
import itertools
import time
import torch
import numpy as np
import random
import os
import json

from model import CAIMVR
from utils.get_mask import get_mask
from utils.util import cal_classify
from utils.logger_ import get_logger
from utils.datasets_supervised import *
from configure.configure_supervised import get_default_config
import collections
import warnings
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")

dataset = {
    0: "Caltech101-7",
    1: "hand",
    2: "NoisyMNIST",
    3: "LandUse_21",
    4: "3Sources",
    5: "Scene_15",
    6: "Caltech101-20",
    7: "BBCSport"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='3', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--eval_num', type=int, default='50', help='gap of print losses')
parser.add_argument('--print_num', type=int, default='50', help='gap of print losses')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments
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
    logger.info('Dataset: ' + str(dataset))

    # Load data
    np.random.seed(1)
    random.seed(1)
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    label_raw = Y_list[0]

    # Create directories
    HISTORIQUES_FOLDER = "historiques_supervised"
    RESULTS_FOLDER = "results_supervised"
    os.makedirs(HISTORIQUES_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    back = 0

    # Log configuration
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    fold_acc, fold_precision, fold_f_measure,fold_f_auc = [], [], [], []
    results_combination = []

    # Multiple runs with different seeds
    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        # Split data into train/test
        len1 = x1_train_raw.shape[1]
        data = np.concatenate([x1_train_raw, x2_train_raw], axis=1)
        x_train, x_test, labels_train, labels_test = train_test_split(
            data, label_raw, test_size=0.2, random_state=data_seed
        )

        # Separate views
        x1_train = x_train[:, :len1]
        x2_train = x_train[:, len1:]
        x1_test = x_test[:, :len1]
        x2_test = x_test[:, len1:]

        # Get masks for train and test
        mask_train = get_mask(2, x1_train.shape[0], config['missing_rate'])
        x1_train = x1_train * mask_train[:, 0][:, np.newaxis]
        x2_train = x2_train * mask_train[:, 1][:, np.newaxis]

        mask_test = get_mask(2, x1_test.shape[0], config['missing_rate'])
        x1_test = x1_test * mask_test[:, 0][:, np.newaxis]
        x2_test = x2_test * mask_test[:, 1][:, np.newaxis]

        # Convert to tensors
        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        mask_train = torch.from_numpy(mask_train).long().to(device)
        x1_test = torch.from_numpy(x1_test).float().to(device)
        x2_test = torch.from_numpy(x2_test).float().to(device)
        mask_test = torch.from_numpy(mask_test).long().to(device)

        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)

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
                CAIMVR_model.txt2img.parameters()
            ),
            lr=config['training']['lr']
        )

        # Print model info
        logger.info(CAIMVR_model.autoencoder1)
        logger.info(CAIMVR_model.img2txt)
        logger.info(optimizer)

        # Move models to device
        CAIMVR_model.autoencoder1.to(device)
        CAIMVR_model.autoencoder2.to(device)
        CAIMVR_model.img2txt.to(device)
        CAIMVR_model.txt2img.to(device)

        # Training
        acc, precision, f_measure,auc ,results,epoch_pos  = CAIMVR_model.train_supervised(
            config, logger, accumulated_metrics,
            x1_train, x2_train, x1_test, x2_test,
            labels_train, labels_test,
            mask_train, mask_test,
            optimizer, device,data_seed
        )

        fold_acc.append(acc)
        fold_precision.append(precision)
        fold_f_measure.append(f_measure)
        fold_f_auc.append(auc)

    logger.info('--------------------Training over--------------------')

    # Calculate mean and std
    acc_mean, precision_mean, f_measure_mean,auc_mean = cal_classify(
        fold_acc, fold_precision, fold_f_measure,fold_f_auc
    )


if __name__ == '__main__':

    main()
