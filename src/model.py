import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics

from .config import *

class Network(nn.Module):
    def __init__(self, base_model, dropout: float, output_dims: List[int]):
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
        # layers.append(nn.Softmax())

        self.base_model.classifier = nn.Sequential(*layers)
    
    # def forward(self, x):
    #     return self.base_model(x)

    def forward(self, x):
        logits = self.base_model(x)
        return F.log_softmax(logits, dim=1)

# Network(base_model=efficientnet_b4(pretrained=False), dropout=0.3, output_dims=[128, 64, 32])


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
        # return self.model(data.view(-1,))
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
    # def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     loss, output, target = self._common_step(batch, batch_idx)
    #     self.log("val_loss", loss)
    #     return loss
    #     #======
    #     pred = output.argmax(dim=1, keepdim=True)
    #     accuracy = pred.eq(target.view_as(pred)).float().mean()
    #     self.log("val_acc", accuracy, sync_dist=True)
    #     self.log("hp_metric", accuracy, on_step=False, on_epoch=True, sync_dist=True)


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



# def training_step(self, batch, batch_idx):
#     x, y = batch
#     loss, scores, y = self._common_step(batch, batch_idx)

#     self.log_dict(
#         {
#             "train_loss": loss,
#         },
#         on_step=False,
#         on_epoch=True,
#         prog_bar=True,
#     )
    
#     if batch_idx % 100 == 0:
#         x = x[:8]
#         grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
#         self.logger.experiment.add_image("mnist_images", grid, self.global_step)

#     return {"loss": loss, "scores": scores, "y": y}