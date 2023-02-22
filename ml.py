from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

token_path = Path("token.txt")
token = token_path.read_text().strip()

model_id = "CompVis/stable-diffusion-v1-4"
device = "mps"

pipe = StableDiffusionPipeline.from_pretrained(
  model_id,
  revision="fp16",
  torch_dtype=torch.float16,
)
pipe.enable_attention_slicing()

pipe.to(device)

prompt = "a dog with cat ears and drinking whiskey"

_ = pipe(prompt, num_inference_steps=1)

image = pipe(prompt).images[0]

# image.save(f"mimage.png")

def obtain_image(
  prompt: str,
  *, 
  seed: int | None = None,
  num_inference_steps: int = 50,
  guidance_scale: float = 7.5
) -> Image:
  generator = None if seed is None else torch.Generator()
  print(f"Using device: {pipe.device}")
  image: Image = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
  ).images[0]
  return image

# image = obtain_image(prompt, num_inference_steps=5, seed=1024)