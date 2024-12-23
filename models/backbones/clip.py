import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class CLIP(nn.Module):
    """
    Class for the CLIP Text Encoder module.

    Attributes:

    """
    def __init__(
            self,
            model_name="openai/clip-vit-base-patch32", 
            num_trainable_blocks=0,
            return_global_token=True,
            cache_dir=None, 
        ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = CLIPTextModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.num_trainable_blocks = num_trainable_blocks
        self.return_global_token = return_global_token

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        if num_trainable_blocks > 0:
            for layer in self.model.text_model.encoder.layers[-self.num_trainable_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the CLIP Text Encoder module to return the encoder hidden states of the given sequence.

        Args:
            x (dict): The input sequence to the model after tokenization.

        Returns:
            torch.Tensor: The encoder hidden states of the given sequence, and the pooled output if return_global_token is True.
        """

        outputs = self.model(**x)

        hidden_states = outputs.last_hidden_state[:, 1:]
        pooled_output = outputs.pooler_output

        if self.return_global_token:
            return hidden_states, pooled_output
        
        return hidden_states
