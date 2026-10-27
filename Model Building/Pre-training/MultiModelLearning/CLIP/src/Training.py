import torch
import torch.nn as nn
from src import utils

class Trainer:
    def __init__(self, model, optimizer, device, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (images, captions) in enumerate(self.dataloader):
                images = images.to(self.device)
                captions = captions.to(self.device)

                self.optimizer.zero_grad()
                image_features, text_features = self.model(images, captions)
                loss = utils.contrastive_loss(image_features, text_features)
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.dataloader)}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.dataloader)
            print(f"✅ Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, model, device, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, captions in dataloader:
                images = images.to(device)
                captions = captions.to(device)
                image_features = model.image_encoder(images)
                text_features = model.text_encoder(captions)
                loss = utils.contrastive_loss(image_features, text_features)
                total_loss += loss.item()
        return total_loss / len(dataloader)
