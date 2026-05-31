# CCCD OpenCV Crop Service

HTTP service for Google Apps Script. Apps Script sends an image as base64,
this service detects the CCCD/card rectangle with OpenCV, perspective-crops it,
enhances it lightly, and returns a PNG as base64.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Health check:

```bash
curl http://localhost:8080/health
```

## Deploy Cloud Run

```bash
gcloud run deploy cccd-crop-opencv \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars CROP_API_TOKEN=doi_token_bi_mat
```

After deployment, copy the Cloud Run URL:

```text
https://cccd-crop-opencv-xxxxx.a.run.app
```

Set these Apps Script properties:

```text
CCCD_CROP_API_URL = https://cccd-crop-opencv-xxxxx.a.run.app/crop-cccd
CCCD_CROP_API_TOKEN = doi_token_bi_mat
```
