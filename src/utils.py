from transformers import PreTrainedTokenizerBase, AutoTokenizer


def load_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        'speakleash/Bielik-1.5B-v3',
        use_fast=True,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
        add_pad_token=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.add_special_tokens(
        special_tokens_dict={
            "additional_special_tokens": [
                "<MEANING>",
                "<EXAMPLE>",
            ]
        }
    )

    return tokenizer