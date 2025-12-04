from collections import Counter

import lightning.pytorch as pl
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

EXAMPLE_PATTERN = "%s<MEANING>%s"


def tokenize(tokenizer, texts):
    return tokenizer(
        texts['text'],
        add_special_tokens=True,
        return_tensors=None,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        padding_side='right',
        max_length=256,
    )


class LitClassificationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset_path: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.splits = None
        self.class_weights_train = None

        # Precompute class weights immediately so they're available after init
        try:
            self.class_weights_train = self._compute_class_weights()
        except Exception:
            # If computation fails for any reason (e.g. dataset not accessible),
            # leave class_weights_train as None and allow setup to try again.
            self.class_weights_train = None

    def setup(self, stage: str = None):
        ds = (
            load_dataset(self.dataset_path)
            .filter(lambda x: x['znaczenie wyrazów slangowych'] is not None)
            .map(lambda x: {'text': EXAMPLE_PATTERN % (x['słowo slangowe'][:128], x['znaczenie wyrazów slangowych'][:256]) ,})
            .map(
                lambda x: tokenize(self.tokenizer, x),
                batched=True,
                num_proc=20,
                remove_columns=['text'],
                desc='Tokenizing'
            )
            .rename_column('sentyment', 'targets')
            .select_columns(['input_ids', 'attention_mask', 'targets'])
            .with_format('torch')
        )

        ds_train_devtest = ds['train'].train_test_split(test_size=0.2, seed=42)
        ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)

        self.splits = DatasetDict({
            'train': ds_train_devtest['train'],
            'validation': ds_devtest['train'],
            'test': ds_devtest['test']
        }).with_format('torch')


    def _compute_class_weights(self):
        """
        Load raw dataset train split, filter out entries with missing meaning and compute class weights.
        Returns a torch.FloatTensor of shape (n_classes,).
        """
        ds = load_dataset(self.dataset_path)
        if 'train' not in ds:
            return torch.tensor([], dtype=torch.float)

        train = ds['train']
        labels = []
        # iterate over examples, filter same way as in setup
        for item in train:
            if item.get('znaczenie wyrazów slangowych') is None:
                continue
            # collect 'sentyment' field as int
            val = item.get('sentyment')
            if val is None:
                continue
            labels.append(int(val))

        if len(labels) == 0:
            return torch.tensor([], dtype=torch.float)

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        counts = torch.bincount(labels_tensor)

        n_samples = labels_tensor.size(0)
        n_classes = counts.numel()

        counts_float = counts.float()
        zero_mask = counts_float == 0
        safe_counts = counts_float.clone()
        safe_counts[zero_mask] = 1.0

        class_weights = n_samples / (n_classes * safe_counts)
        class_weights[zero_mask] = 0.0

        return class_weights.float()

    def train_dataloader(self):
        return DataLoader(
            self.splits['train'],
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.splits['validation'],
            batch_size=self.batch_size,
            num_workers=2,
        )