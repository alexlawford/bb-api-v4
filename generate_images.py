import base64
from PIL import Image
from io import BytesIO
import torch
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, AutoPipelineForImage2Image
from pipeline_boardsbot import BoardsBotPipeline
from utilities import Layer
from diffusers.utils import load_image

device = "mps"

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

def setup_pipe(device):

    # Controlnets
    openpose = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    ).to(device)
    scribble = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    ).to(device)
    controlnets = {
        "scribble" : scribble,
        "openpose" : openpose
    }

    # Main Pipeline
    pipeline = BoardsBotPipeline.from_single_file(
        "./weights/dreamshaper_8.safetensors", controlnet=openpose, torch_dtype=torch.float16
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True, final_sigmas_type="sigma_min"
    )
    pipeline.to(device)
    pipeline.load_lora_weights("./weights/inkSketch_V1.5.safetensors")
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

    # Refiner
    refiner = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(device)

    return pipeline, refiner, controlnets

def execute_pipeline(layers, mask_image, pipeline, refiner, controlnets, generator, description):

    image = pipeline(
        layers=layers,
        mask_image=mask_image,
        controlnets=controlnets,
        num_inference_steps=18,
        guidance_scale=7.0,
        generator=generator,
        controlnet_conditioning_scale=0.55,
        cross_attention_kwargs={"scale":0.6}
    ).images[0]

    image = image.convert('L')
    image = image.convert('RGB')
    image = image.resize((1024,1024))

    image = refiner(
        prompt="simple pencil sketch of " + description + ", best quality, high quality",
        negative_prompt="signature, watermark, deformed, blur, artifact",
        image=image,
        guidance_scale=7.5,
        num_inference_steps=45,
        strength=0.4,
        generator=generator
    ).images[0]

    return image

def generate(layers_raw, mask, variation):
    pipeline, refiner, controlnets = setup_pipe(device)
    mask_image = decode_base64_image(mask)

    layers = [
        Layer(
            ip_adapter_image=load_image('./images/background.png'),
            prompt="plain white background",
            negative_prompt="people, person, character, man, woman, thing"         
        )
    ]

    description = "simple sketch of "

    for _, layer in enumerate(layers_raw):

        description = description + ", " + layer["prompt"]

        if(layer["type"] == "openpose"):
            prompt = "simple sketch of " + layer["prompt"] + ", plain background, bold outlines, strong lines, best quality, high quality, candid, documentary, indirect gaze"
        else:
            prompt = "simple sketch of " + layer["prompt"] +  ", plain background, bold outlines, strong lines, best quality, high quality"

        layers.append(Layer(
            prompt = prompt,
            control_image=decode_base64_image(layer["control"]),
            ip_adapter_image=load_image("./images/" + layer["ipa_image"] + ".png"),
            controlnet_name=layer["type"],
            negative_prompt="motion blur, dof, dark, dim, blurred, low resolution, low quality, multiple, strip, 2-tone, two tone, looking at viewer, looking forward, necklace, hat, necktie, big boobs, busty, earings, accessories, bra, underwear, bikini, topless, breasts, nude, naked, nsfw, porn, complex, small details, chest, straps, bag, backpack, open shirt, lowres, bad anatomy, worst quality, low quality, blurred, focus, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
        ))    
    
    return execute_pipeline(
        layers = layers,
        mask_image = mask_image,
        pipeline = pipeline,
        refiner = refiner,
        controlnets = controlnets,
        device = device,
        generator = torch.Generator(device=device).manual_seed(variation),
        description = description
    )
