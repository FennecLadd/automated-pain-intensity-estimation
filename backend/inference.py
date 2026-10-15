from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .model import CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def _crop_face(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return frame, False
    x,y,w,h = max(faces, key=lambda z:z[2]*z[3])
    pad = int(0.18*max(w,h))
    x1=max(0,x-pad); y1=max(0,y-pad); x2=min(frame.shape[1],x+w+pad); y2=min(frame.shape[0],y+h+pad)
    return frame[y1:y2,x1:x2], True

@torch.inference_mode()
def predict_video(video_path, model, max_frames=96, target_fps=4.0):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration=frame_count/fps if fps>0 and frame_count>0 else 0.0
    step=max(1,int(round(fps/max(target_fps,0.1))))
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    frames=[]; times=[]; face_hits=0; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%step==0:
            crop,found=_crop_face(frame,detector)
            face_hits+=int(found)
            rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
            frames.append(TRANSFORM(Image.fromarray(rgb)))
            times.append(idx/fps if fps else 0.0)
            if len(frames)>=max_frames: break
        idx+=1
    cap.release()
    if not frames:
        raise RuntimeError("No readable frames were extracted.")
    batch=torch.stack(frames).to(DEVICE)
    model=model.to(DEVICE).eval()
    logits=model(batch)
    probs=torch.softmax(logits,dim=1).cpu().numpy()
    mean_probs=probs.mean(axis=0)
    pred=int(mean_probs.argmax())
    trajectory=[]
    for t,p in zip(times,probs):
        trajectory.append({"time":round(float(t),3),"label":CLASS_NAMES[int(p.argmax())],"confidence":round(float(p.max())*100,2),"probabilities":[round(float(x)*100,2) for x in p]})
    return {
        "frames_analyzed":len(frames),
        "sampled_fps":round(fps/step,2) if fps else target_fps,
        "duration_seconds":round(duration,3),
        "prediction":{"label":CLASS_NAMES[pred],"confidence":round(float(mean_probs[pred])*100,2),"probabilities":[round(float(x)*100,2) for x in mean_probs]},
        "trajectory":trajectory,
        "face_detection_rate":round(face_hits/max(len(frames),1)*100,2)
    }
