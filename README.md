# Multimodal Fake News Detection System  
### Comparing LLMs Across Text + Video Modalities  
**Artificial Intelligence Foundations – ICA 2025**  

---

## 📌 Overview  
This project implements a **multimodal fake news detection system** that uses modern Large Language Models (LLMs) to classify *real vs fake* content across text and video modalities.  

The system serves two purposes:

1. **Interactive Artefact**  
   Users can select text or video samples and request predictions from different LLMs.

2. **Evaluation Framework**  
   Enables fair, systematic comparison of multiple LLMs (Gemini, GPT-4o, Mistral) using real datasets:
   - **LIAR dataset** (text)
   - **FaceForensics++ sampled dataset** (video)

This project supports the ICA requirement of developing an implementation *and* conducting a performance evaluation using modern AI techniques.

---

## 🎯 Objectives  

### **A) Functional Artefact**
- Provide an interactive system where users:
  - Browse random fake/real news samples
  - View metadata and extracted frames
  - Choose an LLM
  - Receive a prediction + explanation

### **B) Research/Evaluation Component**
- Compare multiple LLMs on:
  - Fake news classification accuracy
  - Explanation quality
  - Latency (processing time)
  - Performance across modalities (text + video)
- Report findings in ICA submission

---

## 🧠 Core Features (Implemented)
- ✔ Clean and scalable backend architecture (Flask + modular packages)  
- ✔ Dynamic frame extraction using OpenCV + FFmpeg  
- ✔ Video metadata integration (~120 curated real/fake samples)  
- ✔ LIAR dataset text loader  
- ✔ Gemini 1.5/2.5 Vision API integrated (fully functional)  
- ✔ API endpoints for:
  - `/api/videos/random`
  - `/api/videos/predict/<id>?model=gemini`
- ✔ Temporary session folder for frame caching  
- ✔ Readable structured JSON output  
- ✔ Extensive testing (`tests/gemini_test.py`)  

---

## 🏗️ Project Directory Structure  
```
llms_fake_news_comparison/
│
├── README.md
├── .env
├── llm_venv/
│
├── data/
│   ├── raw/
│   │   ├── text/
│   │   │   └── liar_dataset/
│   │   │       ├── train.tsv
│   │   │       ├── test.tsv
│   │   │       ├── valid.tsv
│   │   │       └── README
│   │   │
│   │   └── video/
│   │       ├── faceforensics/
│   │       │   └── *.mp4
│   │       ├── fake_video_metadata.xlsx
│   │       └── original_video_metadata.xlsx
│   │
│   ├── metadata/
│   │   └── llm_evaluation_metadata.xlsx
│   │
│   ├── samples/
│   │   ├── cleaned_response.txt
│   │   └── extracted_text.json
│   │
│   └── outputs/
│
├── server/
│   ├── app.py
│   │
│   ├── llms/
│   │   ├── base.py
│   │   ├── gemini_wrapper.py
│   │   ├── mistral_wrapper.py
│   │   └── openai_wrapper.py
│   │
│   ├── preprocessing/
│   │   ├── preprocess_video.py
│   │   └── preprocess_text.py
│   │
│   ├── utils/
│   │   ├── random_sampler.py
│   │   ├── deepfake_loader.py
│   │   ├── liar_loader.py
│   │   ├── filepaths.py
│   │   ├── dataset.py
│   │   └── helpers.py
│   │
│   └── temp/
│       └── frames/
│
├── client/
│   ├── terminal_client.py
│   └── client_gui.py
│
├── scripts/
│   ├── evaluate_models.py
│   ├── sample_faceforensics.py
│   └── extract_frames.sh
│
└── tests/
    ├── gemini_test.py
    └── __init__.py
```

---

## ⚙️ Installation

### **1. Clone the repository**
```bash
git clone https://github.com/<your_username>/llms_fake_news_comparison.git
cd llms_fake_news_comparison
```
### **2. Create and activate your own virtual environment**
```bash
python -m venv llm_venv
llm_venv\Scripts\activate  # (Windows)
```
### **3. Install requirements in your own virtual environment**
```bash 
pip install -r requirements.txt
```
### **3. Create and add your models' API keys**
```bash
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=optional
MISTRAL_API_KEY=optional
```

## 🔥 Running the Flask Server
```bash
python server/app.py (# server runs on [text](http://127.0.0.1:5000))
```

## 🔌 API Usage

### 🔹 Get random videos
#### GET /api/videos/random?batch=10

### 🔹 Predict fake/real for a specific video
#### GET /api/videos/predict/<video_id>?model=gemini

## 🙏 Acknowledgements

Datasets used:

- LIAR Dataset — William Wang (2017)

- FaceForensics++ — Rössler et al. (2019)

LLMs/APIs:

- Google Gemini (Generative AI)

- OpenAI GPT Models

- Mistral AI Models

This project forms part of the coursework for:
Artificial Intelligence Foundations, Semester 2, 2025

## 📄 License

#### MIT License — free for academic use.