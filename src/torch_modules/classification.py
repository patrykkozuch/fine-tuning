import torch
import torch.nn as nn
from nlp_agh.transformer.dataset import prepare_mask


class ClassificationTransformer(nn.Module):
    def __init__(self, base_transformer: nn.Module, vocab_size: int, num_classes: int):
        super().__init__()
        self.base_transformer = base_transformer
        self.classifier = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Get the output from the base transformer
        attention_mask = prepare_mask(mask)
        transformer_output = self.base_transformer(x, attention_mask)  # (batch, seq_len, d_model)

        sequence_lengths = mask.sum(dim=1)
        batch_indices = torch.arange(transformer_output.size(0), device=transformer_output.device)
        last_token_indices = (sequence_lengths - 1)
        print(sequence_lengths)

        # Use the output of the last token for classification
        last_token_output = transformer_output[batch_indices, last_token_indices, :]  # (batch, d_model)

        # Pass through the classifier
        logits = self.classifier(last_token_output)  # (batch, num_classes)

        return logits
