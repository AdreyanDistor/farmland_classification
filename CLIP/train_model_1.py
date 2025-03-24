import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import cv2
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random

#use postional embeding
#use this website https://learnopencv.com/clip-model/

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class FarmlandDataset(Dataset):
    def __init__(self, farmland_dir, not_farmland_dir, processor, transform=None, val_split=0.2, is_train=True):
        self.farmland_dir = farmland_dir
        self.not_farmland_dir = not_farmland_dir
        self.processor = processor
        self.transform = transform
        self.image_paths = []
        self.labels = []

        farmland_images = [os.path.join(farmland_dir, filename) for filename in os.listdir(farmland_dir)
                          if filename.endswith((".jpg", ".png", ".tiff"))]
        self.image_paths.extend(farmland_images)
        self.labels.extend([1] * len(farmland_images))

        not_farmland_images = [os.path.join(not_farmland_dir, filename) for filename in os.listdir(not_farmland_dir)
                              if filename.endswith((".jpg", ".png", ".tiff"))]
        self.image_paths.extend(not_farmland_images)
        self.labels.extend([0] * len(not_farmland_images))

        self.balance_dataset()

        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

        split_idx = int(len(self.image_paths) * (1 - val_split))
        if is_train:
            self.image_paths = self.image_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.labels = self.labels[split_idx:]

        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Farmland images: {self.labels.count(1)}")
        print(f"Not Farmland images: {self.labels.count(0)}")

    def balance_dataset(self):
        num_farmland = self.labels.count(1)
        num_not_farmland = self.labels.count(0)

        if num_farmland == num_not_farmland:
            return

        if num_farmland > num_not_farmland:
            minority_label = 0
            minority_size = num_not_farmland
            majority_label = 1
        else:
            minority_label = 1
            minority_size = num_farmland
            majority_label = 0

        majority_indices = [i for i, label in enumerate(self.labels) if label == majority_label]
        sampled_majority_indices = random.sample(majority_indices, minority_size)

        minority_indices = [i for i, label in enumerate(self.labels) if label == minority_label]
        balanced_indices = minority_indices + sampled_majority_indices

        self.image_paths = [self.image_paths[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]

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

    progress_bar = tqdm(dataloader, desc="Validation", unit="batch", mininterval=0.1)

    with torch.no_grad():
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
                "Loss": running_loss / (progress_bar.n + 1),
                "Accuracy": correct_preds / total_preds if total_preds > 0 else 0.0
            })

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    return avg_loss, accuracy

def main():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    batch_size = 2
    farmland_dir = 'seg_images'
    notfarmland_dir = 'seg_images_parks'
    
    train_dataset = FarmlandDataset(
        farmland_dir= farmland_dir,
        not_farmland_dir= notfarmland_dir,
        processor=processor,
        transform=transform,
        is_train=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = FarmlandDataset(
        farmland_dir= farmland_dir,
        not_farmland_dir= notfarmland_dir,
        processor=processor,
        transform=transform,
        is_train=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-6)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn, device)
        val_loss, val_accuracy = validate(model, val_dataloader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "binary_park_clip_model_1.pth")

if __name__ == "__main__":
    main()