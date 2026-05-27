import io
import pickle
import os
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import tritonclient.grpc as grpcclient
from PIL import Image

# -----------------------------
# НАСТРОЙКИ (под вашу модель)
# -----------------------------
TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
CLASSES_PATH = "classes.pkl"
INPUT_SIZE = 64   # 64x64 пикселей

# Загружаем названия классов
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
else:
    classes = ["chicken", "cow", "horse", "sheep"]  # запасной вариант

# Предобработка изображения: ресайз, нормализация /255.0
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)   # (64,64,3) значения 0-255
    img_array = img_array / 255.0                 # нормализация в [0,1]
    # Добавляем batch dimension
    batch = np.expand_dims(img_array, axis=0)     # (1,64,64,3)
    return batch

app = FastAPI(title="Image Classifier API", description="Classification of animals (chicken, cow, horse, sheep) via Triton")

# Глобальные переменные для клиента Triton
triton_client = None
input_name = None
output_name = None

import asyncio

@app.on_event("startup")
async def startup():
    global triton_client, input_name, output_name
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    # Ожидаем готовности Triton
    for attempt in range(30):
        try:
            if triton_client.is_server_live() and triton_client.is_model_ready(MODEL_NAME):
                metadata = triton_client.get_model_metadata(MODEL_NAME)
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
                print(f"Подключено к Triton. Input: {input_name}, Output: {output_name}")
                return
        except Exception as e:
            print(f"Попытка {attempt+1}: {e}")
        await asyncio.sleep(2)
    raise RuntimeError("Не удалось подключиться к Triton")

@app.on_event("shutdown")
async def shutdown():
    if triton_client:
        triton_client.close()

@app.get("/health")
async def health():
    if triton_client is None:
        return {"status": "unhealthy", "reason": "triton client not initialized"}
    try:
        return {
            "status": "healthy",
            "triton_live": triton_client.is_server_live(),
            "model_ready": triton_client.is_model_ready(MODEL_NAME)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/classes")
async def get_classes():
    return {"classes": classes, "num_classes": len(classes)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает изображение (jpg, png) и возвращает предсказанный класс и вероятности.
    """
    try:
        contents = await file.read()
        batch = preprocess_image(contents)   # (1,64,64,3)

        # Отправляем в Triton
        inputs = [grpcclient.InferInput(input_name, batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch)
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        result = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        probabilities = result.as_numpy(output_name)[0]   # (4,)

        predicted_index = int(np.argmax(probabilities))
        predicted_class = classes[predicted_index]
        confidence = float(np.max(probabilities))

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                classes[i]: float(probabilities[i]) for i in range(len(classes))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))