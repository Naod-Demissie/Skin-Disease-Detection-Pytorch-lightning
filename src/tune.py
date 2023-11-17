
import argparse
import os
from typing import List
from typing import Optional

import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

from torchvision.models import (efficientnet_b0, EfficientNet_B0_Weights,
                                efficientnet_b1, EfficientNet_B1_Weights, 
                                efficientnet_b2, EfficientNet_B2_Weights, 
                                efficientnet_b3, EfficientNet_B3_Weights, 
                                efficientnet_b4, EfficientNet_B4_Weights, 
                                efficientnet_v2_m, EfficientNet_V2_M_Weights, 
                                efficientnet_v2_s, EfficientNet_V2_S_Weights)


# src: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_ddp.py
# src: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py


"""
You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch_lightning_simple.py [--pruning]
"""


def objective(trial: optuna.trial.Trial) -> float:
    num_units = trial.suggest_int("NUM_UNITS", 16, 32)
    dropout_rate = trial.suggest_float("DROPOUT_RATE", 0.1, 0.2)
    optimizer = trial.suggest_categorical("OPTIMIZER", ["sgd", "adam"])

    accuracy = train_test_model(num_units, dropout_rate, optimizer)  # type: ignore
    return accuracy


tensorboard_callback = TensorBoardCallback("logs/", metric_name="accuracy")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, timeout=600, callbacks=[tensorboard_callback])





if version.parse(pl.__version__) < version.parse("1.6.0"):
    raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")



def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def objective(trial: optuna.trial.Trial) -> float:




def objective(trial: optuna.trial.Trial) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
    callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        devices=2,
        callbacks=[callback],
        strategy="ddp_spawn",
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    callback.check_pruned()

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning distributed data-parallel training example."
    )
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name="pl_ddp",
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))