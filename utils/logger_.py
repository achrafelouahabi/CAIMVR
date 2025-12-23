import logging
import datetime
import os

def get_logger(config,mode='clustering', optuna=False, trial=None):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    # Use a named logger (avoid root logger)
    logger = logging.getLogger("CAIMVR")


    # ✅ Always compute base_name first
    base_name = f"{config['view']} views_{mode}_{config['dataset']}_{str(config['missing_rate']).replace('.','')}"

    # Add handlers only once
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if optuna:
            filename = f'./logs/{config["view"]} views_{mode}_optuna_trial_{str(trial.number)}_{time_str}.log'
            
        else:
            filename = f'./logs/{base_name}_{time_str}.log'
        os.makedirs('./logs', exist_ok=True)

        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        # Stop messages from also propagating to the root logger
        logger.propagate = False

    # ✅ base_name is now always defined
    return logger, base_name
