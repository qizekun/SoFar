import os
import torch
import torch.nn as nn
from orientation.clip import clip


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        clip = load_clip_to_cpu(text_encoder)

        for p in clip.parameters():
            p.requires_grad = False

        self.transformer = clip.transformer
        self.positional_embedding = clip.positional_embedding
        self.token_embedding = clip.token_embedding
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.dtype = clip.dtype
        self.embed_dim = self.transformer.width

    def forward(self, text):
        prompt_text = clip.tokenize(text, context_length=77).to(device=next(self.parameters()).device)

        b, _ = prompt_text.shape
        x = self.token_embedding(prompt_text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        text_feature = x[torch.arange(x.shape[0]), prompt_text.argmax(dim=-1)] @ self.text_projection
        return text_feature
