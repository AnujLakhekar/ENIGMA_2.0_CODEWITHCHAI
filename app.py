import streamlit as st
import mne
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import tempfile
import os

# --- APP CONFIG ---
st.set_page_config(page_title="MindGuard AI", layout="wide")
st.title("🧠 MindGuard: Schizophrenia Biomarker Analysis")
st.markdown("Early-stage detection using real-time EEG spectral analysis.")

# --- 1. RESEARCH DATASET PIPELINE (Deliverable) ---
st.sidebar.header("📂 Data Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload Patient EEG (.edf, .fif)", type=["edf", "fif"])

col1, col2 = st.columns([2, 1])

if uploaded_file:
    # Handle File Buffering for MNE
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Load Data
        if uploaded_file.name.lower().endswith('.edf'):
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)

        # Standardize channel names for 10-20 Mapping
        raw.rename_channels(lambda x: x.replace('EEG ', '').replace('-Ref', '').strip())
        
        # Apply the 10-20 Montage for Spatial Visualization
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        # --- 2. EEG ANALYSIS INTERFACE (Deliverable) ---
        raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
        
        with col1:
            st.subheader("📊 EEG Signal Stream")
            selected_ch = st.selectbox("Focus Channel", raw.ch_names)
            
            # Plot Time-Series
            data, times = raw[raw.ch_names.index(selected_ch), :1500]
            fig_line = px.line(x=times, y=data[0]*1e6, labels={'x':'Time (s)', 'y':'Amplitude (μV)'})
            fig_line.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_line, use_container_width=True)

            # --- 3. EXPLAINABLE BRAIN VISUALIZATION (Deliverable) ---
            st.subheader("📍 Brain Activity Heatmap (XAI)")
            psds = raw.compute_psd(fmin=1, fmax=40, verbose=False)
            psd_data, freqs = psds.get_data(return_freqs=True)
            mean_power = np.mean(psd_data, axis=1) 

            # Create Topomap using Matplotlib
            fig_map, ax = plt.subplots(figsize=(5, 5))
            fig_map.patch.set_facecolor('#0e1117') # Match Streamlit Dark Theme
            mne.viz.plot_topomap(mean_power, raw.info, axes=ax, show=False, contours=0, cmap='Reds')
            st.pyplot(fig_map)
            st.caption("Red areas indicate regions with higher biomarker activity (typically Frontal/Temporal in Schizophrenia).")

        with col2:
            # --- 4. EARLY RISK SCORING (Deliverable: Real-Time) ---
            st.subheader("🎯 Real-Time Risk Assessment")
            
            # DAR Calculation Logic (Delta-Alpha Ratio)
            delta_mask = (freqs >= 1) & (freqs <= 4)
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            
            delta_power = psd_data[:, delta_mask].mean()
            alpha_power = psd_data[:, alpha_mask].mean()
            
            # Clinical Ratio: Higher Delta/Alpha usually correlates with neural slowing/Schizophrenia
            dar_index = delta_power / (alpha_power + 1e-10) 
            calculated_risk = min(int((dar_index / 3.0) * 100), 100)
            
            st.metric(label="Neuro-Anomaly Score", value=f"{calculated_risk}%", delta=f"{dar_index:.2f} DAR Index")
            
            if calculated_risk > 70:
                st.error("🚨 HIGH RISK: Significant biomarkers detected.")
            elif calculated_risk > 40:
                st.warning("⚠️ MODERATE RISK: Spectral anomalies observed.")
            else:
                st.success("✅ LOW RISK: Normal spectral distribution.")

            st.markdown(f"""
            **Biomarker Breakdown:**
            - **Delta Power (1-4Hz):** {delta_power:.2e}
            - **Alpha Power (8-12Hz):** {alpha_power:.2e}
            - **Status:** {'Abnormal' if dar_index > 2.0 else 'Stable'}
            """)
            
            st.divider()
            
            # Sensor Mapping (Proof of Technical Pipeline)
            st.subheader("📍 Sensor Mapping")
            fig_sensors = raw.plot_sensors(show_names=True, show=False)
            fig_sensors.patch.set_facecolor('#0e1117')
            st.pyplot(fig_sensors)

    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.info("Check if your EDF file contains standard 10-20 channel names (Fp1, Cz, etc.)")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("👋 Welcome! Please upload an EEG file (.edf or .fif) to generate a risk assessment report.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/EEG_10-20_system_with_standard_electrode_names.svg/440px-EEG_10-20_system_with_standard_electrode_names.svg.png", width=300)

st.divider()
st.caption("MindGuard AI | EEG Schizophrenia Detection Prototype | Hackathon 2026")