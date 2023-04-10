import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, DataLoader
from VGGDecoder import Decoder

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10
input_size = 256


def mix_input():
    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets for each input folder
    folder1_dataset = datasets.ImageFolder(root='folder1', transform=image_transforms)
    folder2_dataset = datasets.ImageFolder(root='folder2', transform=image_transforms)
    folder3_dataset = datasets.ImageFolder(root='folder3', transform=image_transforms)

    # Combine datasets into a single dataset
    all_datasets = ConcatDataset([folder1_dataset, folder2_dataset, folder3_dataset])

    # Create dataloader for the combined dataset
    dataloader = DataLoader(all_datasets, batch_size=batch_size, shuffle=True)
    return dataloader


# Definedataloader
dataloader = mix_input()

# Define decoder model
decoder = Decoder(hidden_size=256)  # replace MyDecoder with your own decoder model
decoder.to(device)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# Train decoder
for epoch in range(num_epochs):
    for i, inputs in enumerate(dataloader):
        inputs = inputs.to(device)

        # Forward pass
        # Load VGG model without fully connected layers
        vgg_model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
        vgg_features = vgg_model.features
        encoded = vgg_features(inputs)
        decoded = decoder(encoded)
        loss = criterion(decoded, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 steps
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}")
