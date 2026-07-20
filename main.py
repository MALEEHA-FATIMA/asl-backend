from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
with open('/Users/maleehafatima/Desktop/asl_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
classes = data['classes']

MODEL_PATH = '/Users/maleehafatima/Desktop/hand_landmarker.task'

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

@app.get("/")
def root():
    return {"status": "ASL Translator API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    result = detector.detect(mp_image)
    
    if not result.hand_landmarks:
        return {"prediction": "", "confidence": 0, "hand_detected": False}
    
    lm = result.hand_landmarks[0]
    row = []
    for point in lm:
        row.extend([point.x, point.y, point.z])
    
    probs = model.predict_proba([row])[0]
    confidence = float(max(probs))
    prediction = classes[int(np.argmax(probs))]
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "hand_detected": True
    }