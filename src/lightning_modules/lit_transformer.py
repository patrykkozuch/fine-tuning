from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from nlp_agh.transformer.dataset import prepare_mask
from nlp_agh.transformer.scheduler import TransformerLRScheduler
from torch import nn

import lightning as L

IGNORE_INDEX = -100


class LitTransformer(L.LightningModule):
    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.cfg = cfg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = TransformerLRScheduler(optimizer, d_model=self.cfg["d_model"], warmup_steps=4000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    def _batch_calculate_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch['attention_mask'][..., :-1]
        mask = prepare_mask(mask)
        inputs = batch['input_ids'][..., :-1]
        targets = batch['input_ids'][..., 1:].clone()
        target_attention = batch['attention_mask'][..., 1:]
        targets[target_attention == 0] = IGNORE_INDEX

        output = self.model(inputs, mask)
        loss = nn.functional.cross_entropy(
            input=output.reshape(-1, output.size(-1)),
            target=targets.reshape(-1),
            label_smoothing=0.1,
            ignore_index=IGNORE_INDEX
        )

        preds = output.argmax(dim=-1)
        return preds, loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        _, loss = self._batch_calculate_loss(batch)
        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        preds, loss = self._batch_calculate_loss(batch)
        self.log_dict({'val_loss': loss}, prog_bar=True)
        return preds
