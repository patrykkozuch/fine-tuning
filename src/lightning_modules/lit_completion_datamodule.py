import lightning.pytorch as pl
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


def tokenize(tokenizer, texts):
    return tokenizer(
        texts['text'],
        add_special_tokens=True,
        return_tensors=None,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        padding_side='right',
        max_length=257,
        return_overflowing_tokens=True,
        stride=32
    )


class LitCompletionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset_path: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.splits = None

    def setup(self, stage: str = None):
        ds = (
            load_dataset('json', data_files=[self.dataset_path], split='train')
            .filter(lambda x: x['meta']['quality'] == 'HIGH', num_proc=20, desc='Filtering by quality')
            .map(lambda x: tokenize(self.tokenizer, x), batched=True, num_proc=20, remove_columns=['meta', 'text'],
                 desc='Tokenizing')
            .remove_columns('overflow_to_sample_mapping')
        )

        ds_train_devtest = ds.train_test_split(test_size=0.01, seed=42)

        self.splits = DatasetDict({
            'train': ds_train_devtest['train'],
            'validation': ds_train_devtest['test']
        }).with_format('torch')

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
