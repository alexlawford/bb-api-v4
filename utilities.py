from typing import Type, Literal, Optional
from PIL import Image

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