import os
import sys
import torch
from torch import nn
from typing import List, Optional
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics

sys.path.append('..')
from src.config import *


class Network(nn.Module):
    def __init__(self, base_model, dropout: float, output_dims: List[int]) -> None:
        super().__init__()

        self.base_model = base_model
        input_dim: int = base_model.classifier[1].in_features

        layers: List[nn.Module] = []
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, NUM_CLASSES))
        layers.append(nn.Softmax(dim=-1))

        self.base_model.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class LightningNetwork(pl.LightningModule):
    def __init__(self, 
                 base_model,
                 dropout: float, 
                 output_dims: List[int],
                 learning_rate: float) -> None:
    
        super().__init__()
        self.model = Network(base_model=base_model, dropout=dropout, output_dims=output_dims)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=NUM_CLASSES)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=NUM_CLASSES)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES)
        self.auc = torchmetrics.AUROC(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def _common_step(self, batch: List[torch.Tensor], batch_idx: int) -> List[torch.Tensor]:
        feature, target = batch
        output = self(feature)
        loss = self.loss_fn(output, target)
        return loss, output, target

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, output, target = self._common_step(batch, batch_idx)
        output_ = torch.argmax(output, dim=-1) 
        target_ = torch.argmax(target, dim=-1)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.accuracy(output_, target_),
                "train_f1_score": self.f1_score(output_, target_),
                "train_recall": self.recall(output_, target_),
                "train_precision": self.precision(output_, target_),
                "train_auc": self.auc(output, target_),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, output, target = self._common_step(batch, batch_idx)
        output_ = torch.argmax(output, dim=-1) 
        target_ = torch.argmax(target, dim=-1)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": self.accuracy(output_, target_),
                "val_f1_score": self.f1_score(output_, target_),
                "val_recall": self.recall(output_, target_),
                "val_precision": self.precision(output_, target_),
                "val_auc": self.auc(output, target_),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, output, target = self._common_step(batch, batch_idx)
        output_ = torch.argmax(output, dim=-1) 
        target_ = torch.argmax(target, dim=-1)
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": self.accuracy(output_, target_),
                "test_f1_score": self.f1_score(output_, target_),
                "test_recall": self.recall(output_, target_),
                "test_precision": self.precision(output_, target_),
                "test_auc": self.auc(output, target_),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def predict_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        feature, _ = batch
        feature = feature.reshape(feature.size(0), -1)
        output = self(feature)
        return output
    
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)
