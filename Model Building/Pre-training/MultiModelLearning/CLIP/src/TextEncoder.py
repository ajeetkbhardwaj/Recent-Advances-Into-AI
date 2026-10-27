import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, trans_width, trans_heads, trans_layers, context_length, embed_dim):
        super().__init__()

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, trans_width)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, trans_width))

        # Transformer
        self.trans_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_width, nhead=trans_heads),
            num_layers=trans_layers
        )

        # Projection
        self.ln_final = nn.LayerNorm(trans_width)
        self.text_projection = nn.Linear(trans_width, embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.text_projection.weight, std=0.02)

    def forward(self, text):
        # 1. Token + positional embeddings
        x = self.token_embedding(text)  # [B, S, W]
        x = x + self.positional_embedding[:x.size(1), :]

        # 2. Transformer encoding
        x = x.permute(1, 0, 2)
        x = self.trans_encoder(x)
        x = x.permute(1, 0, 2)

        # 3. Pooling — pick last non-zero token as EOT
        eot_indices = (text != 0).sum(dim=1) - 1
        x = x[torch.arange(x.size(0)), eot_indices]

        # 4. Projection
        x = self.ln_final(x)
        x = self.text_projection(x)
        return x


if __name__ == "__main__":
    vocab_size = 49408
    trans_width = 512
    trans_heads = 8
    trans_layers = 12
    context_length = 77
    embed_dim = 512

    model = TextEncoder(vocab_size, trans_width, trans_heads, trans_layers, context_length, embed_dim)
    dummy_text = torch.randint(0, vocab_size, (4, context_length))
    output = model(dummy_text)
    print("✅ TextEncoder output shape:", output.shape)
