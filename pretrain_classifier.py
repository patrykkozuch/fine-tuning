import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from nlp_agh.transformer.transformer import Transformer

from src.lightning_modules.lit_completion_datamodule import LitCompletionDataModule
from src.lightning_modules.lit_transformer import LitTransformer
from src.utils import load_tokenizer

torch.set_float32_matmul_precision('medium')

base_cfg = {
    "batch_size": 64,
    "max_len": 256,
    "n_blocks": 4,
    "num_heads": 2,
    "dropout_rate": 0.2,
    "d_model": 128,
    "d_ff": 256,
    "log_freq": 1000,
    "prompt_log_freq": 5000,
    "val_freq": 10000,
    "epoches": 1000,
    "chkpoint_freq": 5000,
    "gradient_accumulation_steps": 1,
    "tokenizer": 'speakleash/Bielik-1.5B-v3',
    "slurm_job_id": os.getenv('SLURM_JOB_ID', 'local_run'),
}


tokenizer = load_tokenizer()
datamodule = LitCompletionDataModule(base_cfg['batch_size'], 'data/plwiki.jsonl.zst', tokenizer)

def main():
    cfg = base_cfg.copy()

    model = Transformer(
        vocab_size=len(tokenizer),
        seq_len=cfg["max_len"],
        n_blocks=cfg["n_blocks"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        d_model=cfg["d_model"],
        dropout_rate=cfg["dropout_rate"]
    )

    classifier = LitTransformer(
        model=model,
        cfg=cfg
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint('checkpoints/', monitor='val_loss', save_top_k=1, mode='min')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='max'
    )

    trainer = Trainer(
        max_epochs=50,
        devices='auto',
        accelerator='auto',
        log_every_n_steps=1,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        callbacks=[early_stopping, lr_monitor, model_checkpoint]
    )
    trainer.fit(model=classifier, datamodule=datamodule)

if __name__ == "__main__":
    main()
