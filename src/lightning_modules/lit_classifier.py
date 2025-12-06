from typing import Any

import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

import lightning as L

from src.torch_modules.classification import ClassificationTransformer

IGNORE_INDEX = -100


class LitClassifier(L.LightningModule):
    def __init__(
        self,
        model: ClassificationTransformer,
        cfg: dict,
        lr: float = 1e-3,
        class_weights: torch.Tensor = None,
        freeze_base_epochs: int = 0,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = lr
        self.class_weights = class_weights
        self.freeze_base_epochs = freeze_base_epochs

        # Epoch-level metrics for more reliable evaluation
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=3, average='weighted')
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=3, average='weighted')
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3)

        # Only freeze if freeze_base_epochs > 0
        if self.freeze_base_epochs > 0:
            self._set_base_transformer_frozen(True)

    def _set_base_transformer_frozen(self, frozen: bool):
        for param in self.model.base_transformer.parameters():
            param.requires_grad = not frozen

    def _compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = batch['attention_mask']
        inputs = batch['input_ids']
        targets = batch['targets']

        outputs = self.model(inputs, mask)

        loss = nn.functional.cross_entropy(
            input=outputs,
            target=targets,
            label_smoothing=0.1,
            weight=self.class_weights.to(self.device)
        )

        preds = outputs.argmax(dim=-1)

        return preds, targets, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.model.base_transformer.parameters(), 'lr': self.lr * 0.1, 'weight_decay': 0.01},
                {'params': self.model.classifier.parameters(), 'lr': self.lr, 'weight_decay': 0.01}
            ],
            lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg['max_epochs'],
            eta_min=self.lr * 1e-2
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        if 0 < self.freeze_base_epochs == self.current_epoch:
            self._set_base_transformer_frozen(False)
            print(f"Epoch {self.current_epoch}: Unfreezing base transformer")

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        preds, targets, loss = self._compute_loss(batch)

        self.train_f1.update(preds, targets)
        self.train_accuracy.update(preds, targets)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_f1', self.train_f1.compute(), prog_bar=True)
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)

        self.train_f1.reset()
        self.train_accuracy.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        preds, targets, loss = self._compute_loss(batch)

        self.val_f1.update(preds, targets)
        self.val_accuracy.update(preds, targets)

        self.log('val_loss', loss, prog_bar=True)
        return preds

    def on_validation_epoch_end(self):
        self.log('val_f1', self.val_f1.compute(), prog_bar=True)
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)

        self.val_f1.reset()
        self.val_accuracy.reset()
