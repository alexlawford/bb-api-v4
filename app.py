from pipeline_boardsbot import BoardsBotPipeline
from diffusers import ControlNetModel
import torch
import random
from diffusers.utils import load_image
from utilities import Layer

# Set Up Pipeline
openpose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
).to("mps")
scribble = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16).to("mps")

pipeline = BoardsBotPipeline.from_single_file(
    "./weights/experimentas_mkiii.safetensors", controlnet=openpose, torch_dtype=torch.float16
)
pipeline.to("mps")
pipeline.load_lora_weights("./weights/inkSketch_V1.5.safetensors")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
generator = torch.Generator(device="mps").manual_seed(3111)
# pipeline.set_ip_adapter_scale(0.75)

# Controlnets
controlnets = {
    "scribble" : scribble,
    "openpose" : openpose
}

# Layers
mask_image = load_image('./images/mask-2-enlarged.png')

layers = [
     Layer(
        ip_adapter_image=load_image('./images/background.png'),
        prompt="grey, plain, empty background",
        negative_prompt="people, person, character, man, woman, thing"
    ),   
    Layer(
        control_image=load_image('./images/open-pose.png'),
        ip_adapter_image=load_image('./images/David_Li.png'),
        prompt="Simple sketch of David Li, walking, wearing dark blue suit, feeling happy, plain background",
        controlnet_name="openpose",
        negative_prompt="monster, multiple, photographs, strip, 2-tone, two tone, looking at viewer, looking forward, necklace, hat, necktie, big boobs, busty, earings, accessories, bra, underwear, bikini, topless, breasts, nude, naked, nsfw, porn, complex, small details, chest, straps, bag, backpack, open shirt, lowres, bad anatomy, worst quality, low quality, blurred, focus, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
    ),
    Layer(
        control_image=load_image('./images/scribble.png'),
        ip_adapter_image=load_image('./images/background.png'),
        prompt="Simple sketch of a dog, plain background",
        controlnet_name="scribble",
        negative_prompt="monster, multiple, photographs, strip, 2-tone, two tone, looking at viewer, looking forward, necklace, hat, necktie, big boobs, busty, earings, accessories, bra, underwear, bikini, topless, breasts, nude, naked, nsfw, porn, complex, small details, chest, straps, bag, backpack, open shirt, lowres, bad anatomy, worst quality, low quality, blurred, focus, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
    )
]

# Prediction
images = pipeline(
    layers=layers,
    mask_image=mask_image,
    controlnets=controlnets,
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=generator,
    controlnet_conditioning_scale=0.75,
    cross_attention_kwargs={"scale":0.55}
).images

images[0].save('output/test0026.png')