import torch
import torch.nn as nn
from nlp_agh.transformer.dataset import prepare_mask


class GenerationTransformer(nn.Module):
    def __init__(self, base_transformer: nn.Module, d_model: int, vocab_size: int):
        super().__init__()
        self.base_transformer = base_transformer
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Get the output from the base transformer
        transformer_output = self.base_transformer(x, mask)  # (batch, seq_len, d_model)
        return self.classifier(transformer_output)  # (batch, seq_len, vocab_size)
