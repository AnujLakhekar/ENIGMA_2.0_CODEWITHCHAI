# 🧠 MindGuard: EEG-Based Schizophrenia Detection

MindGuard is an AI-powered clinical decision support tool designed for the early detection of schizophrenia using real-time EEG spectral analysis.

## 🚀 Key Deliverables
- **EEG Analysis Interface**: Real-time signal processing and Butterworth filtering (1-45Hz).
- **Early Risk Scoring**: Dynamic risk assessment based on the clinically validated **Delta-Alpha Ratio (DAR)**.
- **Explainable AI (XAI)**: Brain activity heatmaps (Topomaps) and Power Spectral Density (PSD) visualizations.
- **Research Dataset Pipeline**: Robust handling of industry-standard `.EDF` and `.FIF` files.

## 🛠️ Tech Stack
- **Language**: Python 3.12
- **Signal Processing**: MNE-Python
- **Dashboard**: Streamlit
- **Visuals**: Plotly & Matplotlib
- **Reporting**: FPDF

## ⚙️ Installation
1. Clone the repo: `git clone https://github.com/yourusername/MindGuard.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`