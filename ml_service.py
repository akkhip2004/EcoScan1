# ml_service.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch, io, os
from torchvision import models, transforms
import gdown  # Add this to requirements.txt

# ---------------------- Configuration ----------------------
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Google Drive file ID for the trained model
MODEL_FILE_ID = "1HAMdSfEP82cyH6rzaSgr_ji_lD4rqMDi"
MODEL_PATH = "waste_classifier_new.pt"

# ---------------------- Download Model if Not Present ----------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ---------------------- Preprocessing ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input
    transforms.ToTensor(),          # Convert to tensor
])

# ---------------------- Load Model ----------------------
model = models.resnet18(weights=None)  # same backbone as training
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# ---------------------- FastAPI App ----------------------
app = FastAPI(title="Waste Classifier API", description="Upload an image and get waste classification")

@app.get("/")
async def root():
    return {"message": "Waste Classifier API is running. Use /docs to test."}

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax(1).item()
            predicted_class = CLASS_NAMES[pred_idx]

        # Return prediction and probabilities
        return {
            "prediction": predicted_class,
            "probabilities": {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs.squeeze())}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
