from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    input_path = "input.jpg"
    output_path = "output.jpg"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(input_path)
    results[0].save(filename=output_path)

    return FileResponse(output_path)
