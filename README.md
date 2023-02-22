# StableDiffusionFastApi
Stable Diffusion test with fastapi


Create `token.txt` file and store the HuggingFace token (https://huggingface.co/)

`source env3.1'/bin/activate`

`pip install`

`uvicorn main:app`

Try it with:

`curl "http://localhost:8000/generate?prompt="Cat riding a horse"" --output test.png`

