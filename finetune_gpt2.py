import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification, GPT2ForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
ds = (
    load_dataset('json', data_files=['polish_youth_slang_classification_filtered_last.json.zst'])
    .class_encode_column('label')
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set the EOS token as the padding token (and BOS token since GPT-2 doesn't have one)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token

# Add special tokens for meaning and example
special_tokens = {'additional_special_tokens': ['<MEANING>', '<EXAMPLE>']}
tokenizer.add_special_tokens(special_tokens)

model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3, problem_type='single_label_classification')

# Resize model embeddings to account for new special tokens
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.eos_token_id

model = model.to(device)

print(f"Tokenizer pad token id: {tokenizer.pad_token_id}")
print(f"Model config pad token id: {tokenizer.eos_token_id}")

def tokenize(tokenizer, texts):
    return tokenizer(
        texts,
        add_special_tokens=False,
        return_tensors=None,
        return_attention_mask=False,
        truncation=True,
        max_length=256,
    )

# Tokenize the dataset
def tokenize_function(item):
    MAX_LEN = 128
    word, meaning, text = tokenize(tokenizer, [item['word'], item['meaning'], item['text']])['input_ids']

    meaning_token_id = tokenizer.convert_tokens_to_ids("<MEANING>")
    example_token_id = tokenizer.convert_tokens_to_ids("<EXAMPLE>")

    # Build input_ids with proper truncation
    input_ids = [tokenizer.bos_token_id]
    input_ids += word
    input_ids += [meaning_token_id]

    # Calculate remaining space for meaning (need to leave room for <EXAMPLE>, text, and EOS)
    remaining = MAX_LEN - len(input_ids) - 3  # -3 for <EXAMPLE>, at least 1 text token, and EOS
    meaning_trunc = meaning[:max(0, remaining)]
    input_ids += meaning_trunc

    input_ids += [example_token_id]

    # Calculate remaining space for text (need to leave room for EOS)
    remaining = MAX_LEN - len(input_ids) - 1  # -1 for EOS
    text_trunc = text[:max(0, remaining)]
    input_ids += text_trunc

    input_ids += [tokenizer.eos_token_id]

    # Truncate if still too long
    input_ids = input_ids[:MAX_LEN]

    # Add padding
    seq_len = len(input_ids)
    padding_len = MAX_LEN - seq_len
    input_ids += [tokenizer.pad_token_id] * padding_len

    attention_mask = [1] * seq_len + [0] * padding_len

    return {
        'input_ids': input_ids,
        'label': item['label'],
        'attention_mask': attention_mask
    }

tokenized_datasets = ds.map(tokenize_function, batched=False)['train'].train_test_split(test_size=0.2, stratify_by_column='label')

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_per_class = f1_score(labels, predictions, average=None)
    return {
        'val_accuracy': accuracy_score(labels, predictions),
        'val_f1': f1_score(labels, predictions, average='weighted'),
        'val_f1_class_0': f1_per_class[0],
        'val_f1_class_1': f1_per_class[1],
        'val_f1_class_2': f1_per_class[2],
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='gpt2-finetuned',
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='gpt2-logs',
    eval_strategy='epoch',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
