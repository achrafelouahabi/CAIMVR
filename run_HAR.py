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
parser.add_argument('--test_time', type=int, default='5', help='number of test times')

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
    
    # Set initial random seed
    np.random.seed(1)
    random.seed(1)
    
    # Create results folders
    HISTORIQUES_FOLDER = "historiques"
    RESULTS_FOLDER = "results"
    os.makedirs(HISTORIQUES_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # Hyperparameter values to test
    heads1_values = [16]
    heads2_values = [4]
    lambda1_values = [0.11]
    lambda2_values = [0.1]
    alpha_values = [10]
    lr_values = [4e-4]
    
    best_performances = []
    back = 0
    
    # Iterate over all hyperparameter combinations
    for lambda1, lambda2, alpha, heads1, heads2, lr in itertools.product(
        lambda1_values, lambda2_values, alpha_values, heads1_values, heads2_values, lr_values):
        
        stop = False
        
        # Update config with current hyperparameters
        config['training']['lambda1'] = lambda1
        config['training']['lambda2'] = lambda2
        config['training']['alpha'] = alpha
        config['training']['lr'] = lr
        config['training']['epoch'] = 500
        config['training']['pretrain_epochs'] = 500
        config['Autoencoder']['heads'] = heads1
        config['Prediction']['heads'] = heads2
        
        # Log configuration
        for (k, v) in config.items():
            if isinstance(v, dict):
                logger.info("%s={" % (k))
                for (g, z) in v.items():
                    logger.info("          %s = %s" % (g, z))
            else:
                logger.info("%s = %s" % (k, v))
        
        combination_name = f"Dataset_{dataset}_lambda1_{lambda1}_lambda2_{lambda2}_alpha_{alpha}_heads1_{heads1}_heads2_{heads2}_lr_{lr}"
        
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
            if data_seed == 1 and back == 0:
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

            # # Save weights from first seed (optional, comment out if not needed)
            # if data_seed == 1:
            #     torch.save({
            #         'autoencoder1': CAIMVR_model.autoencoder1.state_dict(),
            #         'autoencoder2': CAIMVR_model.autoencoder2.state_dict(),
            #         'img2txt': CAIMVR_model.img2txt.state_dict(),
            #         'txt2img': CAIMVR_model.txt2img.state_dict(),
            #         'optimizer': optimizer.state_dict()
            #     }, "weights_seed1.pth")
            #     print(">>> Weights from seed 1 saved in weights_seed1.pth")
            
            # Early stopping based on performance thresholds (customize as needed)
            if config['dataset'] == "DHA" and rgb_depth < 0.70:
                print(f"RGB+Depth ({rgb_depth:.4f}) below threshold 0.70, skipping to next combination.")
                stop = True
                break
            elif config['dataset'] == "UWA30" and rgb_depth < 0.60:
                print(f"RGB+Depth ({rgb_depth:.4f}) below threshold 0.60, skipping to next combination.")
                stop = True
                break
            
            # Save detailed results for this seed
            file_path = f"{HISTORIQUES_FOLDER}/{combination_name}_Test_{data_seed}.json"
            with open(file_path, "w") as file:
                json.dump(results, file, indent=4)
            print(f"Results saved to {file_path}")
            
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
            
            # Store results for this seed
            results_combination.append({
                "data_seed": data_seed,
                "rgb": rgb,
                "depth": depth,
                "rgb_depth": rgb_depth,
                "only_rgb": onlyrgb,
                "only_depth": onlydepth,
                "time_elapsed": f"{int(hours)}:{int(minutes)}:{int(seconds)}",
                "epoch_pos": epoch_pos
            })
        
        print("stop:", stop)
        if stop:
            continue
        
        logger.info('--------------------Training over--------------------')
        logger.info(f'Performance for lambda1={lambda1}, lambda2={lambda2}, alpha={alpha}:')
        
        # Calculate statistics
        cal_HAR(fold_rgb, fold_depth, fold_rgbdepth, fold_onlyrgb, fold_onlydepth)
        
        # Calculate mean values for comparison
        mean_rgb = np.mean(fold_rgb)
        mean_depth = np.mean(fold_depth)
        mean_rgb_depth = np.mean(fold_rgbdepth)
        mean_onlyrgb = np.mean(fold_onlyrgb)
        mean_onlydepth = np.mean(fold_onlydepth)
        
        # Save results for this combination
        with open(f"{RESULTS_FOLDER}/results_{combination_name}.json", "w") as file:
            json.dump(results_combination, file, indent=4)
        
        # Add to global performances
        best_performances.append({
            "lambda1": lambda1,
            "lambda2": lambda2,
            "alpha": alpha,
            "heads1": heads1,
            "heads2": heads2,
            "lr": lr,
            "rgb": mean_rgb,
            "depth": mean_depth,
            "rgb_depth": mean_rgb_depth,
            "only_rgb": mean_onlyrgb,
            "only_depth": mean_onlydepth,
            "epoch_pos": epoch_pos
        })
        
        # Save top performances every 5 iterations
        if back % 5 == 0:
            # Sort by rgb_depth performance (change metric as needed)
            best_performances_sorted = sorted(best_performances, key=lambda x: x['rgb_depth'], reverse=True)[:10]
            print(f"--------------------result {back}:------------- ")
            print(best_performances_sorted)
            print("-" * 50)
            
            with open(f"{RESULTS_FOLDER}/best_performances_{back}.json", "w") as file:
                json.dump(best_performances_sorted, file, indent=4)
        
        back += 1
    
    print("All tests completed. Results saved.")


if __name__ == '__main__':

    main()
