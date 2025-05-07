import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from orientation.models.CrossModal import TextEncoder
from orientation.models.transformer import Group, PatchEmbedding, TransformerEncoder

from .build import MODELS
from orientation.utils.logger import *
from orientation.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


@MODELS.register_module()
class PointSO(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = PatchEmbedding(embed_dim=self.embed_dim, input_channel=6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.text_encoder = TextEncoder(config.text_encoder)
        self.text_dim = self.text_encoder.embed_dim
        self.proj = nn.Linear(self.text_dim, self.embed_dim)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.cls_head_finetune = nn.Linear(self.embed_dim * 2, 3)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys), logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, instruction):

        batch_size, num_points, _ = pts.size()
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        with torch.no_grad():
            text_features = self.text_encoder(instruction)
            text_features = text_features.reshape(batch_size, 1, self.text_dim)

        text_features = self.proj(text_features)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos, text_features)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
