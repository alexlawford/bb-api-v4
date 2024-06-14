from pipeline_boardsbot import BoardsBotPipeline
from diffusers import ControlNetModel
import torch
import random
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)

pipeline = BoardsBotPipeline.from_single_file(
    "./weights/experimentas_mkiii.safetensors", controlnet=controlnet, torch_dtype=torch.float16
)

pipeline.to("mps")

pipeline.load_lora_weights("./weights/inkSketch_V1.5.safetensors")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

generator = torch.Generator(device="mps").manual_seed(33)

pipeline.set_ip_adapter_scale(0.45)

controlnet_image = load_image('./images/pose-8.png')
ip_adapter_image = load_image('./images/David_Li.png')

prompt = "Simple sketch of David Li, standing, feeling sad, plain background"

images = pipeline(
    prompt=prompt, 
    image=controlnet_image,
    ip_adapter_image=ip_adapter_image,
    negative_prompt='monster, turtle, multiple, photographs, strip, 2-tone, two tone, looking at viewer, looking forward, Hindu, necklace, hat, necktie, big boobs, busty, earings, accessories, bra, underwear, bikini, topless, breasts, nude, naked, nsfw, porn, complex, small details, chest, straps, bag, backpack, open shirt, lowres, bad anatomy, worst quality, low quality, blurred, focus, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation', 
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=generator,
    controlnet_conditioning_scale=0.55,
    cross_attention_kwargs={"scale":0.5}
).images

images[0].save('output/test0004.png')

