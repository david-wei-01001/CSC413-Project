import torch
import torchvision
import os

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the generator model
model_name = 'stylegan2-ada'
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=True)
model.to(device)
model.eval()

# Define the image size
image_size = 256

# Define the style mixing parameters
mixing_prob = 0.5
latent_dim = 512

# Define the input styles
styles = []
for i in range(2):
    style_path = f'style{i+1}/'
    style_images = []
    for filename in os.listdir(style_path):
        image = torchvision.io.read_image(os.path.join(style_path, filename))
        image = image.to(device)
        image = image.unsqueeze(0).float()
        image = image / 255.0 * 2.0 - 1.0
        style_images.append(image)
    style_images = torch.cat(style_images, dim=0)
    style = model.get_latent(style_images)
    styles.append(style)

# Perform style mixing
with torch.no_grad():
    z1, z2 = torch.randn(2, 1, latent_dim, device=device)
    w1, w2 = model.style_mix(styles[0], styles[1], mixing_prob=mixing_prob)
    w = torch.lerp(w1, w2, 0.5)
    image = model.synthesize(w, noise_mode='const', force_fp32=True)

# Save the output image
out_path = 'out_image/'
os.makedirs(out_path, exist_ok=True)
image = (image.clamp(-1, 1) + 1) / 2.0 * 255.0
image = image.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()[0]
image = torchvision.transforms.functional.to_pil_image(image)
image.save(os.path.join(out_path, 'output.jpg'))
