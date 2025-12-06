"""
Application FastAPI pour Tri‑intelligent
---------------------------------------

Ce module instancie une application FastAPI qui expose :

- une page d'accueil (`/`) rendant un template Jinja2 présentant le projet,
- une page webcam (`/webcam`) permettant de capturer une image via la webcam et
  d'obtenir une prédiction,
- un endpoint API (`/predict`) acceptant un fichier image et renvoyant la
  classe prédite, le nom du bac correspondant et un code couleur.

Le modèle utilisé est chargé au démarrage depuis le dossier `checkpoints/`. Le
mapping des classes est lu à partir de `class_mapping.json`. Les images
reçues sont converties en tenseurs PyTorch et normalisées avant d'être
passées au modèle.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# Définir les constantes de normalisation ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Localisation des fichiers du modèle et mapping
BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"

# Charge le mapping d'indices vers classes
mapping_path = CKPT_DIR / "class_mapping.json"
if not mapping_path.exists():
    raise RuntimeError(f"Fichier de mapping {mapping_path} manquant. Lancez l'entraînement d'abord.")
with mapping_path.open("r", encoding="utf-8") as f:
    idx2class: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du modèle
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(idx2class))
state_path = CKPT_DIR / "model_resnet18.pth"
if not state_path.exists():
    raise RuntimeError(f"Fichier de modèle {state_path} manquant. Lancez l'entraînement d'abord.")
state = torch.load(state_path, map_location=device)
model.load_state_dict(state)
model.eval().to(device)

# Règles de mapping vers les bacs de tri
BIN_RULES = {
    # associe différents labels à la catégorie de bac correspondante
    "verre": "Verre",
    "glass": "Verre",
    "plastic": "Plastique",
    "plastique": "Plastique",
    "metal": "Métal",
    "can": "Métal",
    "aluminum": "Métal",
    # papiers et cartons
    "papier": "Emballages & papiers",
    "papier_emballage": "Emballages & papiers",
    "cardboard": "Emballages & papiers",
    "paper": "Emballages & papiers",
}
BIN_COLORS = {
    "Verre": "#2ecc71",
    "Plastique": "#f39c12",
    "Métal": "#bdc3c7",
    "Emballages & papiers": "#3498db",
    "Non recyclable": "#7f8c8d",
}


def to_bin(label: str) -> str:
    """Détermine le bac de tri à partir du label prédictif."""
    l = label.lower()
    for key, bin_name in BIN_RULES.items():
        if key in l:
            return bin_name
    return "Non recyclable"


app = FastAPI(title="Tri‑intelligent API")

# Serveur de fichiers statiques et configuration des templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Renvoie la page d'accueil."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request) -> HTMLResponse:
    """Renvoie la page de capture webcam."""
    return templates.TemplateResponse("webcam.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str | float]:
    """Prend en entrée une image et retourne la prédiction."""
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).max().item()
        idx = logits.argmax(1).item()
    label = idx2class[idx]
    bin_name = to_bin(label)
    return {
        "label": label,
        "proba": round(prob, 4),
        "bin": bin_name,
        "bin_color": BIN_COLORS.get(bin_name, "#888888"),
    }