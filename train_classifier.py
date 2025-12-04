import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from nlp_agh.transformer.transformer import Transformer

from src.lightning_modules.lit_classification_datamodule import LitClassificationDataModule
from src.lightning_modules.lit_classifier import LitClassifier
from src.torch_modules.classification import ClassificationTransformer
from src.utils import load_tokenizer

torch.set_float32_matmul_precision('medium')

base_cfg = {
    "batch_size": 256,
    "max_len": 256,
    "n_blocks": 4,
    "num_heads": 4,
    "dropout_rate": 0.2,
    "d_model": 256,
    "d_ff": 512,
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
datamodule = LitClassificationDataModule(base_cfg['batch_size'], 'jziebura/polish_youth_slang_classification', tokenizer)

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

    checkpoint = torch.load('checkpoints_pretraining/epoch=9-step=121470.ckpt')

    new_state_dict = {}
    for param in checkpoint['state_dict'].keys():
        if param.startswith('model.base_transformer.'):
            new_key = param[len('model.base_transformer.'):]
            new_state_dict[new_key] = checkpoint['state_dict'][param]

    model.load_state_dict(new_state_dict)

    classification_model = ClassificationTransformer(
        base_transformer=model,
        d_model=cfg["d_model"],
        num_classes=3,
    )

    classifier = LitClassifier(
        model=classification_model,
        cfg=cfg,
        lr=1e-4,
        class_weights=datamodule.class_weights_train
    )

    print(datamodule.class_weights_train)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model_checkpoint = ModelCheckpoint('checkpoints_classifier/', monitor='val_f1', save_top_k=1, mode='max')
    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=10,
        mode='max'
    )

    trainer = Trainer(
        max_epochs=500,
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
