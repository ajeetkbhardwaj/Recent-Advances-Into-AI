import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.ImageEncoder import ImageEncoder
from src.TextEncoder import TextEncoder
from src.CLIP import CLIP
#from src.Training import Trainer

# -----------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    """
    Main script to initialize and test the CLIP model using synthetic data.
    """

    # Hyperparameters (same across encoders)
    input_res = 224
    patch_size = 16
    width = 512
    layers = 6
    heads = 8
    embed_dim = 256
    vocab_size = 10000
    context_length = 77

    logger.info("Initializing CLIP model...")

    # -------------------------------------------------------
    # Initialize CLIP Model (Image + Text Encoders)
    # -------------------------------------------------------
    image_encoder = ImageEncoder(
        input_res=input_res,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=embed_dim
    )

    text_encoder = TextEncoder(
        vocab_size=vocab_size,
        context_length=context_length,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=embed_dim
    )

    clip_model = CLIP(image_encoder, text_encoder, embed_dim=embed_dim)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device)
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------
    # Synthetic Data Creation
    # -------------------------------------------------------
    batch_size = 8
    dummy_images = torch.randn(batch_size, 3, input_res, input_res).to(device)
    dummy_text = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)

    logger.info(f"Synthetic image batch shape: {dummy_images.shape}")
    logger.info(f"Synthetic text batch shape: {dummy_text.shape}")

    # -------------------------------------------------------
    # Forward Pass through CLIP
    # -------------------------------------------------------
    clip_model.eval()
    with torch.no_grad():
        image_features, text_features, logits_per_image, logits_per_text = clip_model(dummy_images, dummy_text)

    # -------------------------------------------------------
    # Outputs
    # -------------------------------------------------------
    logger.info(f"Image features shape: {image_features.shape}")
    logger.info(f"Text features shape: {text_features.shape}")
    logger.info(f"Logits per image shape: {logits_per_image.shape}")
    logger.info(f"Logits per text shape: {logits_per_text.shape}")

    # Check similarity example
    sim = F.cosine_similarity(image_features[0].unsqueeze(0), text_features[0].unsqueeze(0))
    logger.info(f"Cosine similarity between sample image-text pair: {sim.item():.4f}")

    logger.info("✅ CLIP model forward pass successful on synthetic data!")

# -----------------------------------------------------------
# Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
