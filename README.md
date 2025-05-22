# Cognida
Multi‑Camera Person Search & Tracking

A Python-based system to locate and track a specific person across multiple camera feeds using natural language descriptions, and output a stitched movement path as JSON and visualizations.

# System Architecture(1)
┌───────────────────────┐       ┌────────────────────────────────────┐
│  Frame Extraction     │       │ CLI: run_market.py                 │
│  (frame_extraction.py)│──────▶│  • Interactive description prompt  │
└───────────────────────┘       │  • Outputs market_output.json      │
        │                      └────────────────────────────────────┘
        ▼                                       │
┌──────────────────────────┐                    ▼
│ Detection & CLIP Embedding│──┐               ┌─────────────────────────┐
│ (detection_embedding.py) ├─▶│ API: api.py │ Visualization: visualize.py │
└──────────────────────────┘  │  • POST /search               │
        │                     │  • Returns JSON & snapshots   │
        ▼                     └─────────────────────────┘
┌───────────────────────────────────────┐
│ Person Re‑ID & Linking                │
│ (reid_extraction.py + pipeline.py)    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ Path Stitching & JSON Export          │
│ (pipeline.py)                         │
└───────────────────────────────────────┘

# How the Solution Works(2)

Frame Extraction (optional):

frame_extraction.py splits raw videos into per-camera JPEG frames.

Detection & CLIP Embedding:

detection_embedding.py runs YOLOv8 to detect persons in each frame.

Each person crop is embedded with OpenAI CLIP to enable text-based retrieval.

Person Re‑Identification & Linking:

reid_extraction.py loads a pre-trained OSNet model to extract Re‑ID feature vectors.

pipeline.py matches text embeddings to image embeddings (threshold: TEXT_SIM_THRESHOLD), picks the best initial detection, then links subsequent detections by Re‑ID similarity (REID_SIM_THRESHOLD) and temporal consistency (MAX_TIME_GAP).

Path Stitching & JSON Export:

The pipeline sorts linked detections chronologically, builds a path list, and saves cropped snapshot images in snapshots/.

Outputs a JSON object with:

person_id (assigned identifier)

confidence (CLIP match score)

path: array of sightings (camera, frame, enter_ts, exit_ts, bbox, snapshot)

# Interfaces:

# CLI: run_market.py prompts for a description and writes market_output.json.

# API: api.py exposes a Flask endpoint POST /search for remote integration.

# Visualization: visualize.py animates or plots the person’s movement on a simple coordinate grid derived from camera folder names.

# Installation(3)

# Clone the repo
git clone <repo-url> multi_camera_search
cd multi_camera_search

# (Optional) Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download OSNet Re‑ID weights
mkdir -p models
gdown "https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e" \
      -O models/osnet_x0_25_msmt17.pth

# Sample Usage(4)

CLI

python run_market.py
# Prompts:
#   Enter person description:  a person wearing a red hoodie and white sneakers
# Outputs:
#   • snapshots/0001_...jpg
#   • market_output.json

API
# Start server
python api.py

# In another terminal:
curl -X POST http://localhost:5000/search \
     -H "Content-Type: application/json" \
     -d '{
           "description": "tall man in a yellow raincoat",
           "cameras": ["data/Market1501/cams/c1s1", "data/Market1501/cams/c2s2"]
         }'

# Visualization

python visualize.py
# Prompts:
#   Description:  person with a blue backpack
# Shows animated/static plot of movement path

Feel free to customize thresholds in config.py or swap in your own dataset under data/. For issues or enhancements, open a GitHub issue or PR.
