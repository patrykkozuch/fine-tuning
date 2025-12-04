import torch
import torch.nn as nn
from nlp_agh.transformer.dataset import prepare_mask


class ClassificationTransformer(nn.Module):
    def __init__(self, base_transformer: nn.Module, d_model: int, num_classes: int):
        super().__init__()
        self.base_transformer = base_transformer
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        attention_mask = prepare_mask(mask)
        transformer_output = self.base_transformer(x, attention_mask)

        # Mean pooling
        expanded_mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        masked_output = transformer_output * expanded_mask

        sum_embeddings = masked_output.sum(dim=1)  # (batch, hidden_dim)
        sum_mask = mask.sum(dim=1, keepdim=True)  # (batch, 1)

        hidden_state = sum_embeddings / sum_mask.clamp(min=1e-9)

        logits = self.classifier(hidden_state)
        return logits
