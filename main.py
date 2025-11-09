from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import uvicorn

app = FastAPI()

# Настройка CORS - разрешаем запросы с любого домена (для продакшена лучше указать конкретный домен)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно заменить на ["https://image0.tilda.ws"] для безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    payload = {"inputs": req.prompt}
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    result = response.content
    import base64
    encoded_img = base64.b64encode(result).decode("utf-8")
    return {"image_base64": encoded_img}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


