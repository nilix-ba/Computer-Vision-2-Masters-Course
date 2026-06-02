import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from main.model_resnet18 import SaliencyUNet
from main.dataset import SaliencyDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = "./cv2_project_data/"
OUTPUT_DIR = "./predictions/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize and load model architecture weights
model = SaliencyUNet().to(device)
model.load_state_dict(torch.load("best_saliency_model.pth", map_location=device))
model.eval()

test_dataset = SaliencyDataset(DATA_DIR, "test_images.txt", fixations_txt=False) [cite: 145]

to_pil = transforms.ToPILImage()

print("Generating predictions for test set...")
with torch.no_grad():
    for image, relative_path in test_dataset:
        # Prep tensor dim shape
        image_tensor = image.unsqueeze(0).to(device)
        
        # Run forward pass
        output = model(image_tensor)
        output = output.squeeze(0).cpu() # Squeeze back out tracking matrix dimensions
        
        # Convert back down into image space
        predicted_img = to_pil(output)
        
        # Extract image index ID number from text file paths safely (e.g. "images/test/image-4134.png")
        base_name = os.path.basename(relative_path) # "image-4134.png"
        img_id = base_name.split("-")[1].split(".")[0] # "4134"
        
        # Save exact string name specification requested by instructors [cite: 153]
        predicted_img.save(os.path.join(OUTPUT_DIR, f"prediction-{img_id}.png"))

print("Inference generation completed successfully!")