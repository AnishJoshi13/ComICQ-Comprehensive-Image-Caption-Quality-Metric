# Import the necessary libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess an image
image_path = "testim.jpg"
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")

# Generate image caption
with torch.no_grad():
    caption = model.generate(**inputs)

# Decode and print the caption
caption_text = processor.decode(caption[0], skip_special_tokens=True)
print("Generated Caption:", caption_text)
