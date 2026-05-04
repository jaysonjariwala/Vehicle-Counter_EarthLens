# EarthLens - Vehicle Counter (T5.4)

**SMAI (CS7.403) Assignment 3 - IIIT Hyderabad - 2025-26**
**Course Coordinator: Prof. CV Jawahar**

**Team: EarthLens**
- Jayson JJ
- Padmaja Y
- Gangothri C

A Streamlit web app that detects and counts vehicles from images or video using YOLOv8n (pre-trained on COCO). No custom training needed - the model already detects cars, buses, trucks, bicycles and motorcycles out of the box.

---

## What it does

**Phase 1 - Total Vehicles Count in Image/Video**
Upload an image or short video and get a single number output.
For video, also shows cumulative total vehicles count across all frames.

**Phase 2 - Total Vehicles Count by Category in Image/Video**
Same as Phase 1 but also shows a breakdown by vehicle type:
- Bicycle
- Car
- Two-Wheelers (motorcycle)
- Bus
- Truck

For video, shows both peak frame breakdown and cumulative breakdown.

---

## Detection Settings

Four adjustable settings in the sidebar:
- **Confidence Threshold** (slider, default 0.35)
- **IoU Threshold/NMS** (slider, default 0.45)
- **Image Size** (dropdown: 320/640/1280, default 640)
- **Max Detections** (slider, default 100)

Each setting shows a contextual tip when adjusted.

---

## Project Structure

```
EarthLens_Vehicle-Counter/
├── EarthLens.py        # main Streamlit application
├── IIIT_Logo.png       # fallback logo if website fetch fails
├── requirements.txt    # Python dependencies
└── README.md           # this file
```

---

## How to Run Locally

**Step 1 - Clone the repo**
```bash
git clone https://github.com/<your-username>/EarthLens-Vehicle-Counter.git
cd EarthLens-Vehicle-Counter
```

**Step 2 - Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**Step 3 - Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 - Run the app**
```bash
streamlit run EarthLens.py
```

Browser opens at `http://localhost:8501` automatically.
On first run, `yolov8n.pt` (~6 MB) downloads automatically - internet needed once.

---

## Running on Google Colab (free T4 GPU)

```python
!pip install streamlit ultralytics opencv-python-headless pillow requests -q
!wget -q https://raw.githubusercontent.com/<your-username>/EarthLens-Vehicle-Counter/main/EarthLens.py
!wget -q https://raw.githubusercontent.com/<your-username>/EarthLens-Vehicle-Counter/main/IIIT_Logo.png

# Run with localtunnel to get a public URL
!npm install -g localtunnel -q
!streamlit run EarthLens.py &>/content/logs.txt &
!lt --port 8501
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| Detection | YOLOv8n (ultralytics) |
| Image processing | OpenCV, Pillow |
| Dataset | COCO (pre-trained, no extra download) |

---

## Major Limitations

**(1) Indian Vehicle Types Not Present in COCO Dataset:**
YOLOv8n is pre-trained on COCO which has 80 classes. Our code only picks 5 road vehicle classes: Bicycle, Car, Two-Wheeler, Bus, and Truck. Auto-Rickshaws, Small Tempos, Tempo Travellers, and other LCVs like construction vehicles are not present in COCO, so the model either misses them entirely or sometimes confuses them with a car or truck. Additionally, since YOLOv8n is pre-trained on the COCO dataset which is largely based on Western road conditions, it has limited exposure to Indian vehicle types, Indian traffic patterns, and mixed road environments commonly seen in India.

**(2) Peak Count vs Cumulative Count:**
Peak count shows the maximum vehicles visible at any one moment. Cumulative count shows total detections across the full video. Both are useful but neither gives exact unique vehicle count without a proper object tracker.

---

## Acknowledgements
We sincerely acknowledge the invaluable support, guidance, and assistance received from Prof. C.V. Jawahar, TA Head Sachin, the TAs, and the IT Support team (including Deepak and colleagues) during the course work.
LLMs used: CoPilot & Claude (code scaffolding, debugging). All analysis and evaluation done by team members.

---

## License
MIT
