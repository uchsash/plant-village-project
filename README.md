# Plant Village Disease Classifier

This repository contains an end-to-end computer vision and deep learning project designed to classify 38 types of vegetable leaves and identify potential crop diseases. It serves as a core practical demonstration of my deep learning workflow within my Data Science specialization.

## Live Deployment
The interactive model is fully operational and hosted on Hugging Face Spaces. You can upload leaf images (from the dataset below or any of these type) and test the live predictions here:
https://huggingface.co/spaces/uchsash/plant-disease-classifier

## Used Datased from Kaggle:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

## Repository Structure
- `Plant_Village_Project.ipynb`: Interactive Notebook containing dataset exploration, pipeline creation, and model training and evaluation in PyTorch.
- `app.py`: The deployment engine script mapping application logic to the Hugging Face space interface.
- `plantvillage_cnn.pth`: Serialized model weights and neural network layer states from the completed training run.
- `classes.json`: Dictionary mapping index arrays to targeted botanical names and specific disease classifications.
- `requirements.txt`: Specified library environments and package versions needed to execute the workspace locally.

## Technical Details & Optimization
- **Framework:** PyTorch (with full GPU tensor allocation configured during model execution).
- **Data Engineering:** Implemented batch data normalization, structural feature handling, and custom random data splicing techniques for robust input augmentation.
- **Model Training:** Utilized Convolutional Neural Network (CNN) blocks optimized to classify health states over variable plant foliage profiles.

## 💻 Local Setup & Installation
To pull this repository and run the application locally on your machine, follow these steps:

1. Clone this repository to your local path.
2. Install the necessary packages from the terminal:
   ```bash
   pip install -r requirements.txt
   python app.py
