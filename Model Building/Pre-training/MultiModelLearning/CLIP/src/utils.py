import torch
from torch.nn import functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    # Normalize features
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    # Cosine similarity as logits
    logits = torch.matmul(image_features, text_features.t()) / temperature
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

    # Cross-entropy loss for image-to-text and text-to-image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return (loss_i2t + loss_t2i) / 2