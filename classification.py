import os
import argparse
import requests
from bs4 import BeautifulSoup
import pdfplumber
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import shutil

# Define a directory to save temporary images
TEMP_DIR = "temp_images"

# --- Image Extraction Functions ---
def extract_images_from_url(url):
    """
    Extracts and saves images from a given URL to a temporary directory.
    Returns a list of file paths to the extracted images.
    """
    print(f"Extracting images from URL: {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')

        if not img_tags:
            print("No images found on the webpage.")
            return []

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        image_paths = []
        for i, img in enumerate(tqdm(img_tags, desc="Downloading images")):
            img_url = img.get('src')
            if not img_url:
                continue

            # Handle relative URLs
            if not img_url.startswith(('http', 'https')):
                img_url = requests.compat.urljoin(url, img_url)

            try:
                img_data = requests.get(img_url, stream=True, timeout=10).content
                # Use Pillow to verify the image data
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                file_path = os.path.join(TEMP_DIR, f"image_{i}.jpg")
                img.save(file_path)
                image_paths.append(file_path)
            except Exception as e:
                print(f"  Warning: Could not download or process {img_url}. Error: {e}")

        return image_paths
    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL: {e}")
        return []

def extract_images_from_pdf(pdf_path):
    """
    Extracts and saves images from a given PDF file to a temporary directory.
    Returns a list of file paths to the extracted images.
    """
    print(f"Extracting images from PDF: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    image_paths = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting images from PDF")):
                for img_info in page.images:
                    img_name = f"page_{page_num+1}_img_{img_info['x0']}.png"
                    img_path = os.path.join(TEMP_DIR, img_name)
                    
                    try:
                        # Extract the image data
                        img = page.crop(
                            (img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom'])
                        ).to_image()
                        img.original.save(img_path)
                        image_paths.append(img_path)
                    except Exception as e:
                        print(f"  Warning: Could not extract image from page {page_num+1}. Error: {e}")
        return image_paths
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

# --- Model and Classification Functions ---
# --- Model and Classification Functions ---
def get_model(model_path='best_model.pth'):
    """
    Loads a pretrained ResNet50 model and its trained weights.
    Downloads the model file from a URL if it does not exist locally.
    """
    print("Loading the trained ResNet50 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CORRECTED URL of your model file on Google Drive
    model_url = "https://drive.google.com/uc?export=download&id=1jUQ8UXPNreOVbAjAUZdu9n_lxbg8O30f"
    
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Downloading from Google Drive...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(model_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading model"):
                    if chunk:
                        f.write(chunk)
            print("Download complete!")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            exit()

    # Load a ResNet50 model with pretrained weights from ImageNet
    model = models.resnet50(pretrained=True)
    
    # Replace the final fully connected layer for 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load the trained weights from the .pth file
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit()

    # Set the model to evaluation mode for inference
    model.eval()
    
    # Move the model to the selected device (GPU or CPU)
    model = model.to(device)
    
    return model

def classify_image(image_path, model):
    """
    Classifies a single image using the provided model.
    """
    # Define the image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    labels = ["non-medical", "medical"]

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return labels[predicted.item()]
    
    except Exception as e:
        print(f"  Error classifying image {image_path}: {e}")
        return "error"

# --- Main function to orchestrate the process ---
def main():
    """
    Main function to run the image classification pipeline.
    """
    parser = argparse.ArgumentParser(description="Classify images from a URL or PDF.")
    parser.add_argument("input_source", type=str, help="The URL or path to a PDF file.")
    args = parser.parse_args()

    # Determine if the input is a URL or a PDF file
    if args.input_source.startswith(('http', 'https')):
        image_paths = extract_images_from_url(args.input_source)
    elif args.input_source.lower().endswith('.pdf'):
        image_paths = extract_images_from_pdf(args.input_source)
    else:
        print("Invalid input. Please provide a URL or a path to a PDF file.")
        return

    if not image_paths:
        print("No images to classify. Exiting.")
        return
        
    print(f"\nFound {len(image_paths)} images. Classifying...")
    
    # Load the model once
    model = get_model()

    # Classify each image
    classification_results = {}
    for image_path in tqdm(image_paths, desc="Classifying images"):
        label = classify_image(image_path, model)
        classification_results[image_path] = label

    print("\n--- Classification Results ---")
    for img_path, label in classification_results.items():
        print(f"{os.path.basename(img_path)}: {label}")

    # Clean up temporary directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"\nCleaned up temporary directory '{TEMP_DIR}'.")

if __name__ == "__main__":
    main()