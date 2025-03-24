import os
import torch
from torchvision import transforms
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(model_path, device):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, processor

def preprocess_image(image_path, transform):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if transform:
        image = transform(image)

    return image

def predict(model, processor, image_path, transform, device):
    image = preprocess_image(image_path, transform)

    text = ["Farmland", "Not Farmland"]

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image

    probs = torch.softmax(logits_per_image, dim=-1).squeeze().cpu().numpy()

    predicted_class = text[probs.argmax()]
    predicted_prob = probs.max()

    return predicted_class, predicted_prob

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "binary_park_clip_model_1.pth"
    model, processor = load_model(model_path, device)

    test_dir = "seg_images_parks"

    for filename in os.listdir(test_dir):
        if filename.endswith((".jpg", ".png", ".tiff")):
            image_path = os.path.join(test_dir, filename)

            try:
                predicted_class, prob = predict(model, processor, image_path, transform, device)
                print(f"Image: {filename}, Predicted: {predicted_class}, Probability: {prob:.4f}")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

if __name__ == "__main__":
    main()