from typing import Any

import lightning
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import PreTrainedTokenizerBase

class LogPredictionSamplesCallback(Callback):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, logger: WandbLogger):
        super().__init__()
        self.logger = logger
        self.tokenizer = tokenizer

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
            n = 20
            x, y = batch['input_ids'][:n], batch['targets'][:n]

            texts = self.tokenizer.batch_decode(x, skip_special_tokens=True)
            data = [[text, pred, target] for text, pred, target in zip(texts, outputs[:n], y[:n])]

            self.logger.log_table(key="sample_table", columns=['Text', 'Prediciton', 'Target'], data=data)
