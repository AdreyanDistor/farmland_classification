import torch 
import clip
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

model.to(device)

def load_image(image_path, preprocess, device):
    image = Image.open(image_path)
    return preprocess(images = image, return_tensors = 'pt').to(device)

images_dir = 'seg_images'

img_files = [f for f in os.listdir(images_dir) if f.endswith (('.png', '.tiff'))]

images = [load_image(os.path.join(images_dir, img), processor, device) for img in img_files]


print(images)
clip_labels = ['photo of a farmland', 'photo of not a farm land']

label_tokens = processor(
    text = clip_labels,
    padding = True,
    images = None,
    return_tensors= 'pt'
).to(device)

print(label_tokens['input_ids'][0][:10])