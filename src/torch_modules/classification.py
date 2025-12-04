import torch
import torch.nn as nn
from nlp_agh.transformer.dataset import prepare_mask


class ClassificationTransformer(nn.Module):
    def __init__(self, base_transformer: nn.Module, d_model: int, num_classes: int):
        super().__init__()
        self.base_transformer = base_transformer
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Get the output from the base transformer
        attention_mask = prepare_mask(mask)
        transformer_output = self.base_transformer(x, attention_mask)

        expanded = mask.unsqueeze(-1).expand(transformer_output.size())
        masked_output = transformer_output * expanded
        hidden_state = masked_output.sum(dim=1) / expanded.sum(dim=1)

        # Pass through the classifier
        logits = self.classifier(hidden_state)  # (batch, num_classes)

        return logits
