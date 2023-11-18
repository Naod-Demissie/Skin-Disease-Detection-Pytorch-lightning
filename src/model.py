import torch
from torch import nn
from typing import List, Optional
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics

from .config import *


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
        layers.append(nn.Softmax())

        self.base_model.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class LightningNetwork(pl.LightningModule):
    # def __init__(self, input_size, learning_rate, num_classes)
    def __init__(self, 
                 base_model,
                 dropout: float, 
                 output_dims: List[int]) -> None:
        super().__init__()
        self.model = Network(base_model=base_model, dropout=dropout, output_dims=output_dims)

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

        # TODO: check which one works
        output = self.forward(feature)
        # output = self(feature)

        loss = self.loss_fn(output, target)
        return loss, output, target

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        loss, output, target = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "output": output, "target": target}

    #TODO: check the return value of this method
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, output, target = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "output": output, "target": target}

    
    #TODO: check the return value of this method
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, output, target = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    #TODO: check the return value of this method
    def predict_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        feature, _ = batch
        feature = feature.reshape(feature.size(0), -1)
        scores = self.forward(feature)
        # TODO: check if categorical or sparse labels are needed
        preds = torch.argmax(scores, dim=1)
        return preds
    
    #TODO: type annotation
    def training_epoch_end(self, outputs) -> None:
        outputs = torch.cat([x["output"] for x in outputs])
        targets = torch.cat([x["target"] for x in outputs])
        self.log_dict(
            {
                "train_acc": self.accuracy(outputs, targets),
                "train_f1": self.f1_score(outputs, targets),
                "train_recall": self.recall(outputs, targets),
                "train_precision": self.precision(outputs, targets),
                "train_auc": self.auc(outputs, targets),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    #TODO: type annotation
    def validation_epoch_end(self, outputs) -> None:
        outputs = torch.cat([x["output"] for x in outputs])
        targets = torch.cat([x["target"] for x in outputs])
        self.log_dict(
            {
                "val_acc": self.accuracy(outputs, targets),
                "val_f1": self.f1_score(outputs, targets),
                "val_recall": self.recall(outputs, targets),
                "val_precision": self.precision(outputs, targets),
                "val_auc": self.auc(outputs, targets),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # return optim.Adam(self.parameters(), lr=self.lr)