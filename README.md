# 🫀 Heart Disease Predictor: Intelligent Cardiac Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A full-stack Machine Learning solution for real-time heart disease risk prediction, featuring a dynamic tuning dashboard and a production-grade API.**

---

## 🔗 Live Demo
- **Interactive Dashboard (Streamlit):** [https://suraj-heart-guard.streamlit.app](INSERT_YOUR_STREAMLIT_LINK_HERE)
- **Prediction API (Render):** [https://suraj-heart-api.onrender.com/docs](INSERT_YOUR_RENDER_LINK_HERE)

---

## 📖 Project Overview
HeartGuard AI is designed to assist medical professionals in triaging patients for potential heart disease. Unlike static models, this system provides a **dual-interface solution**:

1.  **For Clinicians (Frontend):** An intuitive **Streamlit Dashboard** allowing for patient data entry, visualization of risk factors, and real-time hyperparameter tuning ("Model Doctor").
2.  **For Developers (Backend):** A high-performance **FastAPI endpoint** deployed via Docker, serving the optimized K-Nearest Neighbors (KNN) model for integration into hospital systems.

The core algorithm achieves **~91% accuracy** (at K=7) on the UCI Heart Disease dataset, outperforming standard baseline models.

---

## 🚀 Key Features

### 1. 🏥 Prediction Clinic
- User-friendly sliders for inputting 13 clinical features (Age, Cholesterol, Thalach, etc.).
- Real-time classification (Healthy vs. Disease) with confidence probability scores.
- "Traffic Light" system: Green for Healthy, Red for High Risk.

### 2. ⚙️ The "Model Doctor" (Dynamic Tuning)
- **Interactive Hyperparameter Tuning:** Users can adjust the `K` value (Number of Neighbors) in real-time.
- **Visual Analytics:** Live plotting of Accuracy vs. K-Value to demonstrate model stability.
- **Auto-Recommendation:** The system automatically flags `K=7` as the optimal setting for this dataset.

### 3. 📊 Data Analytics Dashboard
- Exploratory Data Analysis (EDA) charts integrated directly into the app.
- Correlation heatmaps and distribution plots to understand underlying risk factors.

---

## 🛠️ Tech Stack & Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model** | Scikit-Learn | K-Nearest Neighbors (KNN) with StandardScaler. |
| **Frontend** | Streamlit | Interactive web interface & visualization (Seaborn/Matplotlib). |
| **Backend** | FastAPI | High-speed REST API for inference. |
| **Container** | Docker | Containerized deployment for consistent runtime. |
| **Hosting** | Render & Streamlit Cloud | CI/CD automated deployment. |

---

## 💻 Installation & Local Usage

To run this project on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/surajkr214/heart-disease-predictor.git](https://github.com/surajkr214/heart-disease-predictor.git)
cd heart-disease-predictor

## 🔗 Live Demo
- **Interactive Dashboard (Streamlit):** [https://suraj-heart-guard.streamlit.app](https://suraj-heart-guard.streamlit.app)
- **Prediction API (Render):** [https://suraj-heart-api.onrender.com/docs](https://suraj-heart-api.onrender.com/docs)
