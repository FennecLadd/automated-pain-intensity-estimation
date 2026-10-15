# PAIN-AI — Contactless Pain Assessment

A deployable research prototype for contactless pain assessment from facial video.

## Current implemented model
The deployment is prepared for the project's Stage 09B ResNet18 + Focal Loss classifier:
- 4 classes: No Pain, Low Pain, Moderate Pain, High Pain
- Current documented validation results:
  - Accuracy: 78.43%
  - Balanced Accuracy: 56.59%
  - Macro-F1: 59.44%
- Evaluation uses subject-wise separation.
- EmotionPain dataset: 48,398 frames, 25 subjects, 200 sequences.

Do not present these validation metrics as live-video performance. They are research evaluation results.

## Architecture

Browser
  -> FastAPI
  -> video decoding / frame sampling
  -> optional face detection
  -> ResNet18 classifier
  -> frame probabilities
  -> temporal aggregation
  -> dashboard JSON
  -> charts and textual interpretation

The backend deliberately reports PSPI as unavailable unless a separately trained PSPI regression checkpoint is supplied. It does not invent PSPI values.

## Project layout

backend/
  main.py
  model.py
  inference.py
  schemas.py
frontend/
  index.html
  styles.css
  app.js
models/
  resnet18_stage09b_weights.pth      <-- copy your real checkpoint here
  model_config.json                 <-- optional
data/
  project_results.json              <-- optional analytics generated from notebook
requirements.txt
Dockerfile
render.yaml
start.sh
.gitignore

## Local Windows / Linux

Python 3.10–3.12 is recommended.

pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000

Open:
http://127.0.0.1:8000

## Put your model here

Copy the actual trained checkpoint from Colab/Drive:

models/resnet18_stage09b_weights.pth

If your checkpoint has a different filename, set:
MODEL_CHECKPOINT=path/to/checkpoint.pth

If the checkpoint was saved as a full training dictionary, the loader handles common:
- state_dict
- model_state_dict
- model
- raw state_dict

## Optional analytics

Create data/project_results.json using the helper script or export your notebook metrics into it.

The frontend automatically loads:
- class distribution
- confusion matrix
- ordinal error
- PSPI analysis
- confidence distribution
- subject accuracy
- training history

## Important research limitation

The current live inference model is a frame classifier. It should not be described as a clinically validated pain detector. The LSTM/LNN/LTC and BioVid physiological-fusion components remain extension points until their trained checkpoints and experiments are completed.

## Deployment

### Render
The included render.yaml and Dockerfile can be used for deployment.

### Docker
docker build -t pain-ai .
docker run -p 8000:8000 pain-ai

For a large model file, use Git LFS or object storage rather than committing multi-hundred-MB checkpoints directly to Git.

## Webcam

Webcam access requires browser permission. In production it should be served over HTTPS. localhost is also treated as a secure context by modern browsers.

## Medical disclaimer

This is an academic research prototype. Predictions are not medical diagnoses and should not replace professional clinical assessment.
