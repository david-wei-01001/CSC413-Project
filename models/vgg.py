import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


# Set batch size and input image size
batch_size = 16
input_size = 256


def get_data():
    # Load VGG model without fully connected layers
    vgg_model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
    vgg_features = vgg_model.features

    # Set model to evaluation mode
    vgg_model.eval()

    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load input images from folder and preprocess them
    input_folder = 'input'
    input_images = []
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image = image_transforms(image)
        input_images.append(image)

    # Stack input images into a batch tensor
    input_tensor = torch.stack(input_images, dim=0)

    # Pass input tensor through VGG model to get output feature maps
    processed = vgg_features(input_tensor)

    dataset = TensorDataset(processed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
