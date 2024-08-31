# Trains `bigRSNN` model on all sessions of both monkeys.
# # # # # # # # # 

# Pretraining is switched on by default and performed on all sessions of each monkey.
# If pretraining should be limited to the three sessions used in the challenge, run with data.pretrain_filenames=challenge-data


# CONFIG
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
import os
from pathlib import Path

# NUMERIC
from challenge import get_model, train_validate_model, evaluate_model, configure_model
from challenge import get_dataloader
from challenge.utils import save_model_state, load_model_state
from challenge.utils.plotting import plot_training, plot_cumulative_mse
from challenge.utils.misc import convert_np_float_to_float
import torch
import numpy as np

# LOGGING
import logging
import json

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("train-bigRSNN")


@hydra.main(config_path="conf", config_name="train-bigRSNN", version_base="1.1")
def train_all(cfg: DictConfig) -> None:

    logger.info("Starting new simulation...")

    # SETUP
    # # # # # # # # #

    # convert dtype string to torch dtype
    dtype = getattr(torch, cfg.dtype)

    # SETTING RANDOM SEED
    # # # # # # # # #
    
    if cfg.seed:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        
        if torch.cuda.is_available and 'cuda' in cfg.device:
            torch.cuda.manual_seed(cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        os.environ['PYTHONHASHSEED'] = str(cfg.seed)

    # Get dataloader
    dataloader = get_dataloader(cfg, dtype=dtype)

    # # # # # # # # #

    for monkey_name in ["loco", "indy"]:

        nb_inputs = cfg.data.nb_inputs[monkey_name]

        # PRETRAINING OR LOADING STATE DICT
        # # # # # # # # # # #
        if cfg.pretraining:
            filenames = list(cfg.data.pretrain_filenames[monkey_name].values())

            # GET MODEL
            # # # # # # # # #
            logger.info("Constructing model for " + monkey_name + " pretraining...")
            pretrain_dat, pretrain_val_dat, _ = dataloader.get_multiple_sessions_data(
                filenames
            )
            model = get_model(cfg, nb_inputs=nb_inputs, dtype=dtype, data=pretrain_dat)

            logger.info("Configuring model...")
            model = configure_model(model, cfg, pretrain_dat, dtype)
            
            logger.info("Pretraining on all {} sessions...".format(monkey_name))
            model, history = train_validate_model(
                model,
                cfg,
                pretrain_dat,
                pretrain_val_dat,
                cfg.training.nb_epochs_pretrain,
                verbose=cfg.training.verbose,
                snapshot_prefix="bigRSNN_pretrain_" + monkey_name + "_",
            )
            results = {}
            for k, v in history.items():
                if "val" in k:
                    results[k] = v.tolist()
                else:
                    results["train_" + k] = v.tolist()

            logger.info("Pretraining complete.")

            # Save to JSON file with indentation
            converted_results = convert_np_float_to_float(results)
            with open("bigRSNN-results-pretraining-" + monkey_name + ".json", "w") as f:
                json.dump(converted_results, f, indent=4)
        
            # Save pretrained model state
            save_model_state(model, "bigRSNN-pretrained-" + monkey_name + ".pth")

            pretrained_model = model.state_dict()

        elif cfg.load_state[monkey_name]:
            logger.info("Loading pretrained model for " + monkey_name)
            pretrained_model = load_model_state(cfg.load_state[monkey_name])
            logger.info("Model state loaded.")

        else:
            logger.info("No pretraining or model state loaded.")
            pretrained_model = None

        # TRAINING & EVALUATION
        # # # # # # # # #

        for session_name, filename in cfg.data.filenames[monkey_name].items():

            logger.info("=" * 50)
            logger.info("Constructing model for " + session_name + "...")
            logger.info("=" * 50)

            train_dat, val_dat, test_dat = dataloader.get_single_session_data(filename)
            model = get_model(cfg, nb_inputs=nb_inputs, dtype=dtype, data=train_dat)

            logger.info("Configuring model...")
            model = configure_model(model, cfg, train_dat, dtype)
            
            # Load pretrained model state
            if pretrained_model is not None:
                model.load_state_dict(pretrained_model)
                logger.info("Pretrained model state loaded.")

            logger.info("Training on " + session_name + "...")
            model, history = train_validate_model(
                model,
                cfg,
                train_dat,
                val_dat,
                cfg.training.nb_epochs_train,
                verbose=cfg.training.verbose,
                snapshot_prefix="bigRSNN_" + session_name + "_",
            )

            results = {}
            for k, v in history.items():
                if "val" in k:
                    results[k] = v.tolist()
                else:
                    results["train_" + k] = v.tolist()

            logger.info("Training complete.")
            

            # SAVE MODEL STATE
            # # # # # # # # #
            
            # Local save in hydra run directory
            save_model_state(model, "bigRSNN-" + session_name + ".pth")

            # If seed is set, save model state as 'models / {session_name} / bigRSNN-{seed}.pth'
            if cfg.seed:
                path = Path(to_absolute_path('models')) / session_name
                path.mkdir(parents=True, exist_ok=True)
                filepath = path / ("bigRSNN-" + str(cfg.seed) + ".pth")
                save_model_state(model, filepath)
                
            logger.info("Saved model state.")


            # EVALUATE MODEL
            # # # # # # # # #
            logger.info("Evaluating model...")

            if cfg.plotting.plot_cumulative_mse:
                fig, ax = plot_cumulative_mse(
                    model, val_dat, save_path="bigRSNN_cumulative_se_" + session_name + ".png"
                )

            model, scores, pred, bm_results = evaluate_model(model, cfg, test_dat)

            logger.info("Benchmark results:")
            for k, v in bm_results.items():
                # log key and value rounded to 4 decimal places
                if isinstance(v, float):
                    logger.info(f"{k}: {v:.4f}")
                else:
                    logger.info(f"{k}: {v}")

            for k, v in model.get_metrics_dict(scores).items():
                results["test_" + k] = v

            # Save to JSON file with indentation
            converted_results = convert_np_float_to_float(results)
            with open("bigRSNN-results-" + session_name + ".json", "w") as f:
                json.dump(converted_results, f, indent=4)

            if cfg.plotting.plot_training:
                fig, ax = plot_training(
                    results,
                    cfg.training.nb_epochs_train,
                    names=["loss", "r2"],
                    save_path="bigRSNN_training_" + session_name + ".png",
                )
                

if __name__ == "__main__":
    train_all()
