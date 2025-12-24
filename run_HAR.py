import argparse
import itertools
import time
import torch
import os
import numpy as np
import random
import json

from model import CAIMVR
from utils.util import cal_HAR
from utils.logger_ import get_logger
from utils.datasets_supervised import data_loader_HAR
from configure.configure_supervised import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

dataset = {
    0: "DHA",
    1: "UWA30",
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--eval_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='1', help='number of test times')

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
    config['print_num'] = args.print_num
    config['eval_num'] = args.print_num
    config['view'] = 3
    config['dataset'] = dataset
    
    logger, plt_name = get_logger(config)
    
    logger.info('Dataset:' + str(dataset))
    
        
    # Log configuration
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    fold_rgb, fold_depth, fold_rgbdepth, fold_onlyrgb, fold_onlydepth = [], [], [], [], []
    results_combination = []

    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        
        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        seed = config['seed'] * data_seed
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Load data
        train_data = data_loader_HAR(config['dataset'])
        train_data.read_train()

        # Build model
        CAIMVR_model = CAIMVR(config)
        
        optimizer = torch.optim.Adam(
            itertools.chain(
                CAIMVR_model.autoencoder1.parameters(), 
                CAIMVR_model.autoencoder2.parameters(),
                CAIMVR_model.img2txt.parameters(), 
                CAIMVR_model.txt2img.parameters()),
            lr=config['training']['lr'])

        # Print the models (only for first seed of first combination)
        if data_seed == 1 :
            logger.info(CAIMVR_model.autoencoder1)
            logger.info(CAIMVR_model.img2txt)
            logger.info(optimizer)

        # Move models to device
        CAIMVR_model.autoencoder1.to(device)
        CAIMVR_model.autoencoder2.to(device)
        CAIMVR_model.img2txt.to(device)
        CAIMVR_model.txt2img.to(device)

        # Training
        rgb, depth, rgb_depth, onlyrgb, onlydepth, results, epoch_pos = CAIMVR_model.train_HAR(
            config, logger, accumulated_metrics, train_data, optimizer, device)


              
        fold_rgb.append(rgb)
        fold_depth.append(depth)
        fold_rgbdepth.append(rgb_depth)
        fold_onlyrgb.append(onlyrgb)
        fold_onlydepth.append(onlydepth)

        # Calculate elapsed time
        duration_seconds = time.time() - start
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"Test N° {int(data_seed)} duration: {int(hours)}:{int(minutes)}:{int(seconds)}")
        

    
    logger.info('--------------------Training over--------------------')
    # Calculate statistics
    cal_HAR(fold_rgb, fold_depth, fold_rgbdepth, fold_onlyrgb, fold_onlydepth)



if __name__ == '__main__':
    main()

