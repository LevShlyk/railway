from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Модель запроса от клиента
class GenerateRequest(BaseModel):
    prompt: str

HUGGINGFACE_API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json",
}

@app.post("/generate")
def generate_image(req: GenerateRequest):
    if not HUGGINGFACE_API_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face API token is not set")

    payload = {
        "inputs": req.prompt,
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.content  # Обычно возвращается изображение в бинарном виде

    # В этом примере вернем изображение в base64 для удобства передачи в JSON
    import base64
    encoded_img = base64.b64encode(result).decode("utf-8")

    return {"image_base64": encoded_img}