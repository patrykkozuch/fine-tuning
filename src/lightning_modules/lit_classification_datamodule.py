import random
from collections import Counter

import lightning.pytorch as pl
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

def tokenize(tokenizer, texts):
    return tokenizer(
        texts,
        add_special_tokens=False,
        return_tensors=None,
        return_attention_mask=False,
        truncation=True,
        max_length=256,
    )


def collate_fn(tokenizer: PreTrainedTokenizerBase, batch: list[dict]):
    final_batch = {'input_ids': [], 'targets': [], 'attention_mask': []}
    for item in batch:
        word, meaning, text = tokenize(tokenizer, [item['word'], item['meaning'], item['text']])['input_ids']

        input_ids = [tokenizer.bos_token_id]
        input_ids += word + [tokenizer.convert_tokens_to_ids("<MEANING>")]
        input_ids += meaning[:128 - len(input_ids) - 2]
        input_ids += [tokenizer.convert_tokens_to_ids("<EXAMPLE>")]
        input_ids += text[:128 - len(input_ids) - 1]
        input_ids += [tokenizer.eos_token_id]
        padding_len = 128 - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_len

        attention_mask = [1] * (128 - padding_len) + [0] * padding_len

        final_batch['input_ids'].append(input_ids)
        final_batch['targets'].append(item['targets'])
        final_batch['attention_mask'].append(attention_mask)

    final_batch = {k: torch.tensor(v, dtype=torch.long) for k, v in final_batch.items()}

    return final_batch


class LitClassificationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset_path: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.splits = self.load_dataset()
        self.class_weights_train = None

        # Precompute class weights immediately so they're available after init
        try:
            self.class_weights_train = self._compute_class_weights('train')
        except Exception:
            self.class_weights_train = None

    def load_dataset(self):
        ds = load_dataset('json', data_files=[self.dataset_path]).rename_column('label', 'targets').class_encode_column('targets')

        # Use stratified split to maintain class distribution
        ds_train_devtest = ds['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column='targets')

        return DatasetDict({
            'train': ds_train_devtest['train'],
            'validation': ds_train_devtest['test'],
        }).with_format('torch')

    def _compute_class_weights(self, split: str = 'train'):
        ds = self.splits[split].to_pandas()
        label_counts = Counter(ds['targets'])
        num_classes = len(label_counts)
        class_weights = [1 / label_counts[i] for i in range(num_classes)]
        return torch.tensor(class_weights, dtype=torch.float) / sum(class_weights)

    def train_dataloader(self):
        return DataLoader(
            self.splits['train'],
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=lambda batch: collate_fn(self.tokenizer, batch)
        )

    def val_dataloader(self):
        return DataLoader(
            self.splits['validation'],
            batch_size=self.batch_size,
            num_workers=2,
            collate_fn=lambda batch: collate_fn(self.tokenizer, batch)
        )
