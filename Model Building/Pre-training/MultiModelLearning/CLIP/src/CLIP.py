import torch
import torch.nn as nn
import numpy as np
from src.ImageEncoder import ImageEncoder
from src.TextEncoder import TextEncoder

class CLIP(nn.Module):
    def __init__(self, embed_dim, img_res, vision_layers, vision_width, vision_heads, vision_patch,
                 context_length, vocab_size, trans_width, trans_heads, trans_layers):
        super().__init__()

        self.image_encoder = ImageEncoder(
            input_res=img_res,
            patch_size=vision_patch,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            trans_width=trans_width,
            trans_heads=trans_heads,
            trans_layers=trans_layers,
            context_length=context_length,
            embed_dim=embed_dim
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        return image_features, text_features