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
        layers.append(nn.Softmax(dim=1))

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
        self.validation_step_outputs: List = []

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=NUM_CLASSES)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=NUM_CLASSES)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES)
        self.auc = torchmetrics.AUROC(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def _common_step(self, batch: List[torch.Tensor], batch_idx: int) -> List[torch.Tensor]:
        feature, target = batch
        output = self.forward(feature)
        loss = self.loss_fn(output, target)
        return loss, output, target

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> dict:
        loss, output, target = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "output": output, "target": target}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> dict:
        loss, output, target = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append(
            {
                "loss": loss, 
                "output": output, 
                "target": target
            }   
        )

        self.log_dict(
            {
                "loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "output": output, "target": target}

    
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, output, target = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        feature, _ = batch
        feature = feature.reshape(feature.size(0), -1)
        output = self.forward(feature)
        # TODO: check if categorical or sparse labels are needed
        preds = torch.argmax(output, dim=1)
        return preds
    
    # def on_train_epoch_end(self, outputs: torch.Tensor) -> None:
    def on_train_epoch_end(self, outputs: List[dict]) -> None:
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

    # def on_validation_epoch_end(self, outputs: List[dict]) -> None:
    # # def on_validation_epoch_end(self, outputs: torch.Tensor) -> None:
    #     outputs = torch.cat([x["output"] for x in outputs])
    #     targets = torch.cat([x["target"] for x in outputs])
    #     self.log_dict(
    #         {
    #             "val_acc": self.accuracy(outputs, targets),
    #             "val_f1": self.f1_score(outputs, targets),
    #             "val_recall": self.recall(outputs, targets),
    #             "val_precision": self.precision(outputs, targets),
    #             "val_auc": self.auc(outputs, targets),
    #         },
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )

    def on_validation_epoch_end(self) -> None:
    # def on_validation_epoch_end(self, outputs: torch.Tensor) -> None:
        outputs = torch.stack([x['output'] for x in self.validation_step_outputs])
        targets = torch.stack([x['target'] for x in self.validation_step_outputs])
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])

        self.log_dict(
            {
                "val_loss": losses.mean(),
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
        return optim.Adam(self.parameters(), lr=self.learning_rate)
