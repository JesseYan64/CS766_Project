import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from dataset import mask
from dataset import transforms as T
from gan.generator import UnetGenerator

def test():
    generator = UnetGenerator()
    generator.load_state_dict(torch.load("./runs/generator.pt"))
    generator.eval()

    transforms = T.Compose([T.Resize((256,256)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])])

    dataset = mask.Mask(root='.', transform=transforms, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for m, _ in dataloader:
        print(m.shape)
        # Convert the tensor to a numpy array and adjust the dimension order for matplotlib
        # The 'm' tensor is expected to be in the format [B, C, H, W], where
        # B = Batch size, C = Channels, H = Height, W = Width
        image_to_plot = m[0].numpy().transpose(1, 2, 0)  # Select the first image in the batch and rearrange dimensions
        # Normalize the image data to [0, 1] for correct visualization
        image_to_plot = (image_to_plot - image_to_plot.min()) / (image_to_plot.max() - image_to_plot.min())
        test_image = Image.fromarray((image_to_plot * 255).astype('uint8'))

        # Prepare your test image
        transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_image = transform(test_image).unsqueeze(0)  # Add a batch dimension

        # Generate the output
        with torch.no_grad():  # No need to track gradients for the test image
            generated_image = generator(test_image).squeeze(0)  # Remove the batch dimension

        # Convert the output tensor to an image
        generated_image = generated_image.mul(0.5).add(0.5)  # Denormalize
        generated_image = generated_image.clamp(0, 1)  # Clamp the values to ensure they're between 0 and 1
        generated_image = generated_image.permute(1, 2, 0)  # Change the tensor shape from CxHxW to HxWxC

        # Display the generated image
        plt.imshow(generated_image.numpy())
        plt.show()

if __name__ == "__main__":
    test()