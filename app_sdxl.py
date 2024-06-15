# from pipeline_boardsbot_sd_xl import StableDiffusionXLBoardsBotPipeline
# from diffusers import ControlNetModel, AutoencoderKL
# from diffusers.utils import load_image
# import numpy as np
# import torch

# pipeline = StableDiffusionXLBoardsBotPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
# ).to("mps")

# prompt = "a robot dog"
# negative_prompt = "low quality, bad quality"

# # control image
# control_image = load_image(
#     "./images/open-pose.png"
# )

# # initialize the models and pipeline
# controlnet_conditioning_scale = 0.5  # recommended for good generalization

# controlnet = ControlNetModel.from_pretrained(
#     "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
# ).to("mps")

# # xinsir/controlnet-scribble-sdxl-1.0

# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("mps")

# pipe = StableDiffusionXLBoardsBotPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
# )

# pipe.enable_model_cpu_offload()

# # generate image
# image = pipe(
#     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=control_image
# ).images[0]

# image.save('output/test_sdxl_001.png')

# !pip install opencv-python transformers accelerate

from pipeline_boardsbot_sd_xl import StableDiffusionXLBoardsBotPipeline

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import torch

from PIL import Image

prompt = "a man in a suit dancing"
negative_prompt = "low quality, bad quality, sketches"

# download an image
control_image = load_image(
    "./images/open-pose.png"
)

# initialize the models and pipeline
controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
).to("mps")

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("mps")

pipe = StableDiffusionXLBoardsBotPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
).to("mps")

# generate image
image = pipe(
    prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=control_image
).images[0]

image.save("output/sdxl_image_0002.png")