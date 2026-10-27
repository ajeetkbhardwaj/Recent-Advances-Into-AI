import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """
    Vision Transformer (ViT)-style Image Encoder for CLIP.

    Args:
        input_res (int): Input image resolution (e.g., 224)
        patch_size (int): Patch size (e.g., 16)
        width (int): Transformer width (embedding dimension per token)
        layers (int): Number of Transformer encoder layers
        heads (int): Number of self-attention heads
        output_dim (int): Output embedding dimension for CLIP contrastive head
    """
    def __init__(self, input_res, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_res = input_res
        self.output_dim = output_dim
        self.patch_size = patch_size

        # 1. Patchify image
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        num_patches = (input_res // patch_size) ** 2
        self.num_patches = num_patches

        # 2. Learnable class and positional embeddings
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, width))

        # 3. Transformer Encoder (multi-layer self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            batch_first=True,      # ✅ ensures shape = [B, Seq, Dim]
            norm_first=True        # ✅ improves training stability
        )
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # 4. Projection head to CLIP embedding space
        self.ln_pre = nn.LayerNorm(width)
        self.fc = nn.Linear(width, output_dim, bias=False)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.fc.weight, std=0.02)
        # Initialize convolution like ViT
        nn.init.xavier_uniform_(self.conv1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of image encoder.
        Args:
            x (torch.Tensor): input images of shape [B, 3, H, W]
        Returns:
            torch.Tensor: encoded image features [B, output_dim]
        """
        # 1. Patchify
        x = self.conv1(x)  # [B, width, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, width]

        # 2. Add [CLS] token
        cls_token = self.class_embedding.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N_patches+1, width]

        # 3. Add positional embeddings
        x = x + self.positional_embedding

        # 4. Transformer encoder
        x = self.ln_pre(x)
        x = self.trans_encoder(x)  # [B, N_patches+1, width]

        # 5. Take the [CLS] token representation and project
        x = self.fc(x[:, 0, :])  # [B, output_dim]

        # 6. Normalize (optional but improves contrastive alignment)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


if __name__ == "__main__":
    model = ImageEncoder(
        input_res=224,
        patch_size=16,
        width=256,
        layers=4,
        heads=8,
        output_dim=512
    )
    dummy_img = torch.randn(2, 3, 224, 224)
    out = model(dummy_img)
    print("✅ Output shape:", out.shape)
    # Expected: torch.Size([2, 512])
