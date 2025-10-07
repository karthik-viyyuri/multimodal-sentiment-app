
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np
import joblib
import faiss
import pickle
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from io import BytesIO

app = FastAPI()

# Load models
text_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mlp_model = joblib.load("/Users/karthikviyyuri/Documents/finalmini/final_multimodal_sentiment_app/mlp_model.pkl")

text_faiss = faiss.read_index("/Users/karthikviyyuri/Documents/finalmini/final_multimodal_sentiment_app/text_faiss.index")
image_faiss = faiss.read_index("/Users/karthikviyyuri/Documents/finalmini/final_multimodal_sentiment_app/image_faiss.index")

with open("/Users/karthikviyyuri/Documents/finalmini/final_multimodal_sentiment_app/text_cache.pkl", "rb") as f:
    text_cache = pickle.load(f)
with open("/Users/karthikviyyuri/Documents/finalmini/final_multimodal_sentiment_app/image_cache.pkl", "rb") as f:
    image_cache = pickle.load(f)

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-text")
def analyze_text(req: TextRequest):
    emb = text_model.encode(req.text, normalize_embeddings=True)
    D, I = text_faiss.search(np.array([emb]), k=1)
    if D[0][0] > 0.90 and I[0][0] in text_cache:
        return {"sentiment": int(text_cache[I[0][0]]), "source": "cache"}
    fused = np.concatenate([emb, np.zeros(512)])
    pred = mlp_model.predict([fused])[0]
    return {"sentiment": int(pred), "source": "mlp"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = (emb[0] / emb[0].norm()).cpu().numpy()
    D, I = image_faiss.search(np.array([emb]), k=1)
    if D[0][0] > 0.90 and I[0][0] in image_cache:
        return {"sentiment": int(image_cache[I[0][0]]), "source": "cache"}
    fused = np.concatenate([np.zeros(768), emb])
    pred = mlp_model.predict([fused])[0]
    return {"sentiment": int(pred), "source": "mlp"}
