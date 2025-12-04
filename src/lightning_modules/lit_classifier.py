from typing import Any

import torch
import torchmetrics.functional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from nlp_agh.transformer.scheduler import TransformerLRScheduler
from torch import nn

import lightning as L

IGNORE_INDEX = -100


class LitClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, cfg: dict, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = TransformerLRScheduler(optimizer, d_model=self.cfg["d_model"], warmup_steps=1000)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            #     "frequency": 1
            # },
        }

    def _batch_calculate_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = batch['attention_mask']
        inputs = batch['input_ids']
        targets = batch['targets']

        output = self.model(inputs, mask)

        loss = nn.functional.cross_entropy(output, targets)

        preds = output.argmax(dim=-1)
        f1 = torchmetrics.functional.f1_score(
            preds,
            targets,
            num_classes=3,
            average='macro',
            task='multiclass'
        )

        accuracy = torchmetrics.functional.accuracy(
            preds,
            targets,
            num_classes=3,
            task='multiclass'
        )

        return preds, loss, f1, accuracy

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        _, loss, f1, accuracy = self._batch_calculate_loss(batch)
        self.log_dict(
            {
                'train_loss': loss,
                'train_f1': f1,
                'train_accuracy': accuracy
            },
            prog_bar=True
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        preds, loss, f1, accuracy = self._batch_calculate_loss(batch)
        self.log_dict(
            {
                'val_loss': loss,
                'val_f1': f1,
                'val_accuracy': accuracy
            },
            prog_bar=True
        )
        return preds
