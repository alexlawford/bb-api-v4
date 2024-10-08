from pipeline_boardsbot import BoardsBotPipeline
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, AutoPipelineForImage2Image
import torch
from diffusers.utils import load_image
from utilities import Layer

# Set Up Pipeline
openpose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
).to("mps")
scribble = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16).to("mps")

model = "./weights/dreamshaper_8.safetensors"

pipeline = BoardsBotPipeline.from_single_file(
    "./weights/dreamshaper_8.safetensors", controlnet=openpose, torch_dtype=torch.float16
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
	pipeline.scheduler.config, use_karras_sigmas=True
)

img2img = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("mps")

pipeline.to("mps")
pipeline.load_lora_weights("./weights/inkSketch_V1.5.safetensors")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
generator = torch.Generator(device="mps").manual_seed(101)
# pipeline.set_ip_adapter_scale(0.75)

# Controlnets
controlnets = {
    "scribble" : scribble,
    "openpose" : openpose
}


people = [
    ("ophelia", "old asian woman"),
    ("pearl", "asian woman"),
    ("quella", "young asian woman"),
    ("rachel", "old African-American woman"),
    ("sarah", "African-American woman"),
    ("talia", "young African-American woman"),
    ("uma", "old caucasian woman"),
    ("veronica", "fat caucasian woman"),
    ("wendy", "young caucasian woman"),
    ("xara", "old middle-eastern woman"),
    ("yasmin", "middle-eastern woman"),
    ("zada", "young middle-eastern woman"),
    ("alex", "old asian man"),
    ("ben", "asian man"),
    ("chris", "young asian man"),
    ("dan", "old African-American man"),
    ("edward", "African-American man"),
    ("frank", "young African-American man"),
    ("gerald", "old caucasian man"),
    ("hector", "caucasian man"),
    ("ian", "young caucasian man"),
    ("james", "old middle-eastern man"),
    ("karl", "middle-eastern man"),
    ("lex", "young middle-eastern man"),
    ("marty", "fat caucasian man"),
    ("naomi", "arabic woman"),
]

c_image = load_image('./images/open-pose.png')

c_image = c_image.resize((768, 768))

# Layers
for name, description in people:

    mask_image = load_image('./images/mask-2-enlarged.png')

    layers = [
        Layer(
            ip_adapter_image=load_image('./images/background.png'),
            prompt="plain white background",
            negative_prompt="people, person, character, man, woman, thing"
        ),   
        Layer(
            control_image=c_image,
            ip_adapter_image=load_image("./characters/" + name + ".png"),
            prompt="simple sketch of " + name + ", " + description + ", walking, feeling content, plain background, bold outlines, strong lines, best quality, high quality, candid, documentary, indirect gaze",
            controlnet_name="openpose",
            negative_prompt="motion blur, dof, dark, dim, blurred, low resolution, low quality, multiple, strip, 2-tone, two tone, looking at viewer, looking forward, necklace, hat, necktie, big boobs, busty, earings, accessories, bra, underwear, bikini, topless, breasts, nude, naked, nsfw, porn, complex, small details, chest, straps, bag, backpack, open shirt, lowres, bad anatomy, worst quality, low quality, blurred, focus, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
        ),
        Layer(
            control_image=load_image('./images/scribble.png'),
            ip_adapter_image=load_image('./images/background.png'),
            prompt="simple sketch of a wolf, plain background, bold outlines, strong lines, best quality, high quality",
            controlnet_name="scribble",
            negative_prompt="motion blur, dof, dark, dim, blurred, low resolution, low quality, multipl, strip, 2-tone, two tone, looking at viewer, looking forward, necklace, hat, necktie, big boobs, busty, earings, accessories, bra, underwear, bikini, topless, breasts, nude, naked, nsfw, porn, complex, small details, chest, straps, bag, backpack, open shirt, lowres, bad anatomy, worst quality, low quality, blurred, focus, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
        )
    ]

    # Prediction
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

    image = img2img(
        prompt="simple pencil sketch of a " + description + ", walking a dog, best quality, high quality",
        negative_prompt="signature, watermark, deformed, blur, artifact",
        image=image,
        guidance_scale=7.5,
        num_inference_steps=45,
        strength=0.4,
        generator=generator
    ).images[0]

    filename = name + "_" + description.replace(" ", "_")

    image.save("output/" + filename + "_002.png")