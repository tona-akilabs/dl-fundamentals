import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

# Load a pretrained generator (you need a trained model or checkpoint)
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

netG = Generator()
#netG.load_state_dict(torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True))  # Load pretrained weights
netG.load_state_dict(torch.load('dcgan_generator.pth'))  # Load pretrained weights
netG.eval()

# Generate image from noise
noise = torch.randn(1, 100, 1, 1)  # random latent vector
fake_image = netG(noise)

# Save the image
save_image(fake_image, 'generated.png', normalize=True)
