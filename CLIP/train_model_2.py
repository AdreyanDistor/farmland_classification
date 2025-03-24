import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import cv2
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

class FarmlandDataset(Dataset):
    def __init__(self, farmland_dir, not_farmland_dir, processor, transform=None):
        self.farmland_dir = farmland_dir
        self.not_farmland_dir = not_farmland_dir
        self.processor = processor
        self.transform = transform
        self.image_paths = []
        self.labels = []

        farmland_images = [os.path.join(farmland_dir, filename) for filename in os.listdir(farmland_dir)
                          if filename.endswith((".jpg", ".png", ".tiff"))]
        if farmland_images:
            selected_farmland_image = random.choice(farmland_images)
            self.image_paths.append(selected_farmland_image)
            self.labels.append(1)

        not_farmland_images = [os.path.join(not_farmland_dir, filename) for filename in os.listdir(not_farmland_dir)
                              if filename.endswith((".jpg", ".png", ".tiff"))]
        self.image_paths.extend(not_farmland_images)
        self.labels.extend([0] * len(not_farmland_images))

        self.balance_dataset()

        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Farmland images: {self.labels.count(1)}")
        print(f"Not Farmland images: {self.labels.count(0)}")

    def balance_dataset(self):
        num_farmland = self.labels.count(1)
        num_not_farmland = self.labels.count(0)

        majority_size = max(num_farmland, num_not_farmland)

        if num_farmland < num_not_farmland:
            farmland_indices = [i for i, label in enumerate(self.labels) if label == 1]
            sampled_indices = random.choices(farmland_indices, k=majority_size - num_farmland)
            self.image_paths.extend([self.image_paths[i] for i in sampled_indices])
            self.labels.extend([1] * (majority_size - num_farmland))

        print(f"Updated Farmland images: {self.labels.count(1)}")
        print(f"Updated Not Farmland images: {self.labels.count(0)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Failed to load image: {self.image_paths[idx]}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            text = "Farmland" if self.labels[idx] == 1 else "Not Farmland"

            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True, do_rescale=False)

            return inputs, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return None

def custom_collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, Any]:
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return {"inputs": [], "labels": torch.tensor([])}

    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    return {"inputs": inputs, "labels": labels}

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch", mininterval=0.1)

    for batch in progress_bar:
        inputs_list = batch["inputs"]
        labels = batch["labels"].to(device)

        if len(inputs_list) == 0:
            continue

        batch_logits = []

        for inputs in inputs_list:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

            batch_logits.append(logits_per_image)

        batch_logits = torch.cat(batch_logits, dim=0)

        batch_logits = batch_logits.squeeze(1)

        loss = loss_fn(batch_logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (torch.sigmoid(batch_logits) > 0.5)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        progress_bar.set_postfix({
            "Loss": running_loss / (progress_bar.n + 1),
            "Accuracy": correct_preds / total_preds if total_preds > 0 else 0.0
        })

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    return avg_loss, accuracy

def validate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", unit="batch", mininterval=0.1)
        for batch in progress_bar:
            inputs_list = batch["inputs"]
            labels = batch["labels"].to(device)

            if len(inputs_list) == 0:
                continue

            batch_logits = []

            for inputs in inputs_list:
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image

                batch_logits.append(logits_per_image)

            batch_logits = torch.cat(batch_logits, dim=0)

            batch_logits = batch_logits.squeeze(1)

            loss = loss_fn(batch_logits, labels.float())
            running_loss += loss.item()

            predicted = (torch.sigmoid(batch_logits) > 0.5)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            progress_bar.set_postfix({
                "Val Loss": running_loss / (progress_bar.n + 1),
                "Val Accuracy": correct_preds / total_preds if total_preds > 0 else 0.0
            })

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    return avg_loss, accuracy

def main():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    full_dataset = FarmlandDataset(
        farmland_dir='seg_images',
        not_farmland_dir='seg_images_parks',
        processor=processor,
        transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')
    patience = 3
    epochs_without_improvement = 0

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = validate(model, val_dataloader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

        scheduler.step()

    torch.save(model.state_dict(), "binary_park_clip_model_2.pth")

if __name__ == "__main__":
    main()