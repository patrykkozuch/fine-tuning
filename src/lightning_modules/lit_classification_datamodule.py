import lightning.pytorch as pl
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

EXAMPLE_PATTERN = "%s<MEANING>%s<EXAMPLE>%s"


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

    def setup(self, stage: str = None):
        ds = (
            load_dataset(self.dataset_path)
            .filter(lambda x: x['znaczenie wyrazów slangowych'] is not None)
            .map(lambda x: {'text': EXAMPLE_PATTERN % (x['słowo slangowe'], x['znaczenie wyrazów slangowych'], x['tekst']) ,})
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