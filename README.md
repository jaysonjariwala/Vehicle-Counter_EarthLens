# EarthLens - Vehicle Counter (T5.4)

**SMAI (CS7.403) - IIIT Hyderabad - 2025-26**

**Team: EarthLens**
- Jayson JJ
- Padmaja Y
- Gangothri CJ

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
Most important limitation. Auto-Rickshaws, Tempos, Tempo Travellers and other LCVs are not in COCO dataset. Model either misses them completely or sometimes confuses them with Car or Truck. Since COCO is mostly Western road data, model has very less exposure to Indian traffic conditions.

**(2) Frame Sampling May Miss Fast Moving Vehicles:** 
For video processing, we sample only 2 frames per second to keep it fast on CPU. Fast moving vehicles like highway traffic may enter and exit between sampled frames and get missed completely.

**(3) Cumulative Count is Not Unique Vehicle Count:**
Cumulative count adds same vehicle multiple times across frames. We used ByteTrack object tracking to estimate unique count but even that is not 100% accurate if vehicle leaves and re-enters the frame.

**(4) Peak Count Depends on Single Frame:**
Peak count shows maximum vehicles visible at one moment in one frame. It depends on camera angle and road width so it cannot be compared across different videos.

**(5) Night Time and Poor Visibility:**
Model is mostly trained on daytime images. Performance drops in night time footage which is common in dashcam videos.

**(6) No Ground Truth Evaluation:**
We did not have labelled dataset for our test videos so we could not calculate standard metrics like Precision, Recall or mAP. Evaluation is only visual/qualitative.

**(7) Model is Not Fine-Tuned for India:**
We used YOLOv8n exactly as it comes without any fine-tuning on Indian traffic data. A fine-tuned model would give much better results.

**(8) Occlusion in Dense Traffic:**
When vehicles overlap each other in frame, YOLO may miss one or merge both into single box. Very common problem in Indian city traffic.

---

## Acknowledgements
We sincerely acknowledge the invaluable support, guidance, and assistance received from Prof. C.V. Jawahar, TA Head Sachin, the TAs, and the IT Support team (including Deepak and colleagues) during the coursework and throughout this project. Their inputs and feedback helped us understand the subject better and guided us in building this application. We thank all for the clear assignment guidelines and their prompt responses during the project.
LLMs used: MS CoPilot and Claude (code scaffolding, debugging, and initial report drafting about Data, Model description). All analysis, evaluation, observations, and testing were done by the team members. 

---

## License
NA
