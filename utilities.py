from typing import Type, Literal, Optional
from PIL import Image
import torch

class Layer:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: Optional[Type[Image.Image]] = None,
        controlnet_name: Optional[Literal['openpose', 'scribble']] = None,
        ip_adapter_image:  Optional[Type[Image.Image]] = None,
    ) -> None:
        self.prompt = prompt    
        self.negative_prompt = negative_prompt
        self.control_image = control_image
        self.controlnet_name = controlnet_name
        self.ip_adapter_image = ip_adapter_image

def soft_clamp_tensor(input_tensor, threshold=3.5, boundary=4):
    if max(abs(input_tensor.max()), abs(input_tensor.min())) < 4:
        return input_tensor
    channel_dim = 1

    max_vals = input_tensor.max(channel_dim, keepdim=True)[0]
    max_replace = ((input_tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
    over_mask = (input_tensor > threshold)

    min_vals = input_tensor.min(channel_dim, keepdim=True)[0]
    min_replace = ((input_tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
    under_mask = (input_tensor < -threshold)

    return torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, input_tensor))

# Center tensor (balance colors)
def center_tensor(input_tensor, channel_shift=1, full_shift=1, channels=[0, 1, 2, 3]):
    for channel in channels:
        input_tensor[0, channel] -= input_tensor[0, channel].mean() * channel_shift
    return input_tensor - input_tensor.mean() * full_shift

# Maximize/normalize tensor
def maximize_tensor(input_tensor, boundary=4, channels=[0, 1, 2]):
    min_val = input_tensor.min()
    max_val = input_tensor.max()

    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    input_tensor[0, channels] *= normalization_factor

    return input_tensor
