
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from io import BytesIO
from sklearn.linear_model import LogisticRegression

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Dummy fallback classifier (placeholder)
mlp_model = LogisticRegression().fit(np.random.rand(10, 1280), np.random.randint(0, 3, size=10))

device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-text")
async def analyze_text(req: TextRequest):
    text_emb = text_model.encode(req.text, normalize_embeddings=True)
    dummy_img_emb = np.zeros(512)
    fused = np.concatenate([text_emb, dummy_img_emb])
    pred = mlp_model.predict([fused])[0]
    return {"sentiment": int(pred)}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = clip_model.get_image_features(**inputs)
    image_emb = (image_emb[0] / image_emb[0].norm()).cpu().numpy()
    dummy_text_emb = np.zeros(768)
    fused = np.concatenate([dummy_text_emb, image_emb])
    pred = mlp_model.predict([fused])[0]
    return {"sentiment": int(pred)}
