# Medical-and-Non-Medical-Image-Classifier

Project Overview
This project implements a machine learning pipeline to automatically classify images as "medical" or "non-medical" from a given input, which can be a website URL or a PDF document. The goal is to evaluate the ability to design and implement a robust classification solution, optimize for accuracy and speed, and document the process effectively.

My Approach: Transfer Learning with ResNet50
I chose a transfer learning approach using a pre-trained ResNet50 model. This is an efficient and effective method for image classification, especially when working with a custom dataset.

Pre-trained Model: The ResNet50 model, a powerful 50-layer Convolutional Neural Network (CNN), was pre-trained on the vast ImageNet dataset. This means it already possesses a strong foundation for recognizing general image features like edges and textures.

Fine-tuning: Instead of training from scratch, I "froze" the initial layers of the ResNet50 model and only trained a new, final layer to classify images into two categories: "medical" or "non-medical." This significantly reduces training time and data requirements while achieving high accuracy.

Model Parameters:

Loss Function: CrossEntropyLoss, a standard choice for classification tasks.

Optimizer: Adam, an adaptive optimizer that efficiently updates model weights.

Activation Function: ReLU (Rectified Linear Unit), a computationally efficient function used throughout the ResNet architecture.

Dataset and Balancing
A crucial part of this project was creating a diverse and unbiased dataset. I combined two types of image data:

Medical Images: I combined all available 2D datasets from the MedMNIST library (e.g., PneumoniaMNIST, BloodMNIST, OrganAMNIST) to create a comprehensive "medical" category. This gives the model a broad understanding of medical imagery.

Non-Medical Images: I used the widely known CIFAR-10 dataset, which contains images of common objects like cars, birds, and animals.

To prevent model bias, I implemented a data balancing strategy. The script identifies the size of the smaller dataset and randomly samples from the larger one to ensure that both classes have an equal number of images for training and validation.

Performance and Results
The model was trained on the balanced dataset, achieving a validation accuracy of approximately 99.97%. This exceptionally high score demonstrates the model's ability to accurately and reliably distinguish between medical and non-medical images.

The use of transfer learning ensures that the final model is not only accurate but also efficient, with a relatively small file size (best_model.pth) and fast inference times.

How to Run the Project
Clone the Repository:

git clone https://github.com/your-username/Medical-Image-Classifier.git
cd Medical-Image-Classifier

Set up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows

Install Dependencies:

pip install -r requirements.txt

Run the Classifier:

To classify images from a URL:

python final_classification_script.py "URL"

To classify images from a PDF:

python final_classification_script.py "Path to pdf document"

This will run the full pipeline, from image extraction to classification, and provide the results in the terminal.
