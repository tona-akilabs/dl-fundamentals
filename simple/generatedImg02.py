from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

image = pipe("a futuristic cityscape with flying cars").images[0]
image.save("output.png")
