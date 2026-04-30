# NeuroSense

AI-Powered Multimodal Stress Detection System

A real-time 5-modality stress assessment system combining computer vision, audio analysis, NLP, EEG biomarkers, and behavioral surveys — delivering results in under 2 seconds with 75–92% accuracy.

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://neurosense-stress-detection-cyfhjyxw7ap5dz8ujvmdqo.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bbharatharatna/neurosense-stress-detection)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## Overview

NeuroSense is an AI-powered 5-modality stress detection system that combines facial expression analysis, voice emotion recognition, text sentiment analysis, EEG brain signal processing, and behavioral survey responses into a unified stress assessment framework. The system extracts 67+ features across all modalities using a confidence-weighted fusion algorithm, delivering real-time stress assessment with personalized wellness recommendations and downloadable PDF reports with historical analytics.

---

## Key Features

- Facial expression analysis using OpenCV (15 features)
- Voice stress detection via audio signal processing (12 features)
- Text sentiment analysis with negation-aware NLP (13 features)
- EEG brain signal classification using Random Forest (17 biomarkers)
- Behavioral wellness survey with exponential weighting (3 factors)
- Confidence-weighted fusion across all 5 modalities
- Personalized stress recommendations based on level and profile
- PDF report generation with historical analytics
- CSV and JSON export for assessment history
- 100% local processing — no external APIs, privacy-first

---

## Modality Breakdown

| Modality            | Features    | Method                                          |
|---------------------|-------------|-------------------------------------------------|
| Facial Expression   | 15 features | OpenCV — symmetry, texture, edges, brightness   |
| Voice Emotion       | 12 features | Pitch autocorrelation, energy, ZCR, pauses      |
| Text Sentiment      | 13 features | Negation-aware NLP, complexity, punctuation     |
| EEG Brain Signals   | 17 features | Band powers, Alpha/Beta ratio, Hjorth, entropy  |
| Behavioral Survey   | 3 factors   | Sleep, workload, mood with exponential scaling  |

---

## Tech Stack

| Layer              | Technology                         |
|--------------------|------------------------------------|
| Frontend / UI      | Streamlit                          |
| Computer Vision    | OpenCV                             |
| Machine Learning   | Scikit-learn, Random Forest        |
| Audio Processing   | NumPy, SciPy, st-audiorec          |
| EEG Processing     | NumPy, SciPy                       |
| PDF Generation     | FPDF                               |
| Language           | Python 3.9+                        |

---

## Performance

| Metric                  | Value         |
|-------------------------|---------------|
| Accuracy Range          | 75 – 92%      |
| Response Time           | Under 2 sec   |
| Total Features          | 67+           |
| Supported Modalities    | 5             |

---

## Live Demo

https://neurosense-stress-detection-cyfhjyxw7ap5dz8ujvmdqo.streamlit.app/

---

## Getting Started

### Installation

    git clone https://github.com/bbharatharatna/neurosense-stress-detection.git
    cd neurosense-stress-detection
    pip install -r requirements.txt
    streamlit run app.py

### EEG Setup (Optional)

    Download emotions.csv from Kaggle:
    https://www.kaggle.com/datasets/prashantgehlot2404/eeg-dataset-stress-detection
    Upload it in the EEG Setup tab inside the app to activate EEG analysis.

---

## Usage

1. Open the app using the Live Demo link or run locally
2. Capture your face using the camera input
3. Record 5 to 10 seconds of natural speech
4. Type 50 or more words describing your emotional state
5. Set the lifestyle survey sliders for sleep, workload, and mood
6. Optionally upload EEG data in the EEG Setup tab
7. Click Analyse Stress Level to get full results
8. Download your PDF report from the History tab

---

## How It Works

    +-------------------------------+
    |   5 Modality Inputs           |
    |   Face, Voice, Text,          |
    |   Survey, EEG                 |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Feature Extraction           |
    |  67+ features total           |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Per-Modality Confidence      |
    |  Scoring                      |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Confidence-Weighted Fusion   |
    |  Algorithm                    |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Unified Stress Score         |
    |  Low / Moderate / High        |
    +-------------------------------+
                   |
                   v
    +-------------------------------+
    |  Recommendations + PDF Report |
    +-------------------------------+

---

## Known Limitations

- EEG analysis requires manual upload of the Kaggle emotions.csv dataset.
- Voice recording requires the streamlit-audiorec package to be installed.
- First load may take additional time as dependencies are initialized.

---

## Developer

B Bharatha Ratna

---

This project is for educational and portfolio purposes.
