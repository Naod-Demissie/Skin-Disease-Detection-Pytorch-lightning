import os
import sys
import argparse
from typing import List, Optional

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl

import torch
import torch.nn.functional as F

sys.path.append('..')
from src.model import LightningNetwork
from src.dataset import DataModule
from src.config import *


# src: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_ddp.py
# src: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py


"""
You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch_lightning_simple.py [--pruning]
"""

base_model_names = [
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
    'efficientnet_b4', 'efficientnet_v2_s', 'efficientnet_v2_m'
]

#TODO: output_dim list

def objective(trial: optuna.trial.Trial) -> float:
    # output_dims = [
    #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    # ]

    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    output_dims = trial.suggest_categorical('output_dim', output_dims)
    # lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True
    base_model_name = trial.suggest_categorical('base_model_name', base_model_names)
    base_model = get_base_model(base_model_name)

    model = LightningNetwork(
        base_model=base_model, dropout=dropout, output_dims=output_dims
    )
    datamodule = DataModule(model_name=base_model_name)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=NUM_EPOCHS,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        #TODO: check what the no implies
        devices=1, #2
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_acc")
        ],
        strategy="ddp_spawn",
    )
    
    hyperparameters = dict(base_model_name=base_model_name, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    # callback.check_pruned()
    # validation_accuracy
    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning example."
    )
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    tensorboard_callback = TensorBoardCallback("logs/", metric_name="accuracy")

    study = optuna.create_study(
        #TODO: change the study name
        study_name="pl_ddp",
        storage="sqlite:///example.db",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100, timeout=600, callbacks=[tensorboard_callback])

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# if version.parse(pl.__version__) < version.parse("1.6.0"):
#     raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")
