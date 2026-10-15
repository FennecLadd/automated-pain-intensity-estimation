from pathlib import Path
import json
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from .model import load_model
from .inference import predict_video, DEVICE

ROOT=Path(__file__).resolve().parent.parent
FRONTEND=ROOT/"frontend"
DATA=ROOT/"data"
MODEL_PATH=ROOT/os.getenv("MODEL_CHECKPOINT","models/resnet18_stage09b_weights.pth")
app=FastAPI(title="PAIN-AI Research API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
MODEL=None
MODEL_MESSAGE="Model not loaded."

@app.on_event("startup")
def startup():
    global MODEL,MODEL_MESSAGE
    MODEL,MODEL_MESSAGE=load_model(MODEL_PATH)

@app.get("/api/health")
def health():
    return {"status":"ok","device":str(DEVICE),"model_loaded":MODEL is not None,"model_message":MODEL_MESSAGE}

@app.get("/api/results")
def results():
    p=DATA/"project_results.json"
    if p.exists():
        try: return JSONResponse(json.loads(p.read_text()))
        except Exception: pass
    return JSONResponse({"metrics":{"accuracy":78.43,"balanced_accuracy":56.59,"macro_f1":59.44},"dataset":{"frames":48398,"subjects":25,"sequences":200},"source":"Current documented Stage 09B results; detailed arrays unavailable until exported from notebook."})

@app.post("/api/predict")
async def predict(file: UploadFile=File(...)):
    global MODEL
    if MODEL is None:
        raise HTTPException(503,detail=MODEL_MESSAGE)
    suffix=Path(file.filename or ".mp4").suffix.lower()
    if suffix not in {".mp4",".avi",".mov",".mkv",".webm",".m4v"}: suffix=".mp4"
    try:
        content=await file.read()
        if len(content)>250*1024*1024: raise HTTPException(413,detail="Video exceeds 250 MB limit.")
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
            tmp.write(content); path=Path(tmp.name)
        try:
            out=predict_video(path,MODEL)
        finally:
            path.unlink(missing_ok=True)
        out["model"]="ResNet18 Stage 09B"
        out["pspi_available"]=False
        out["pspi_message"]="PSPI is not predicted by the current 4-class classifier. Add a trained PSPI regression head/checkpoint before reporting PSPI values."
        out["notes"]=[
            "Video output is an inference demonstration, not clinical validation.",
            "Frame probabilities are temporally aggregated for the displayed estimate.",
            "Face detection is optional; frames without a detected face fall back to the full frame.",
            "For temporal research claims, connect the completed LSTM/LNN-LTC checkpoint here."
        ]
        return out
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500,detail=f"Inference failed: {type(e).__name__}: {e}")

@app.get("/")
def index(): return FileResponse(FRONTEND/"index.html")
@app.get("/styles.css")
def css(): return FileResponse(FRONTEND/"styles.css",media_type="text/css")
@app.get("/app.js")
def js(): return FileResponse(FRONTEND/"app.js",media_type="application/javascript")
