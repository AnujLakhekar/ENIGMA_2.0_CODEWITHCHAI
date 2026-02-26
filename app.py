import streamlit as st
import mne
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import tempfile
import os
import time
import pandas as pd
from fpdf import FPDF

# --- APP CONFIG ---
st.set_page_config(page_title="Schizophrenia", page_icon="🧠", layout="wide")
st.title("🧠 Schizophrenia Batch Analysis")

# --- 1. BATCH DATA PIPELINE ---
st.sidebar.header("📂 Batch Pipeline")
uploaded_files = st.sidebar.file_uploader(
    "Upload Patient EEGs (.edf, .fif)", 
    type=["edf", "fif"], 
    accept_multiple_files=True
)

st.sidebar.divider()
st.sidebar.header("🕹️ Simulation Control")
live_stream = st.sidebar.toggle("Simulate Live Analysis Stream")
stream_speed = st.sidebar.select_slider("Stream Speed", options=["Slow", "Normal", "Fast"], value="Slow")

col1, col2 = st.columns([2, 1])

if uploaded_files:
    batch_dar_scores = []
    patient_records = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # Load Data
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False) if uploaded_file.name.endswith('.edf') else mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)

            # --- ROBUST SPATIAL OVERLAP FIX (CHB-MIT SUPPORT) ---
            
            montage_1020 = mne.channels.make_standard_montage('standard_1020')
            std_names_low = {n.lower(): n for n in montage_1020.ch_names}
            
            new_names = {}
            ch_types = {}
            seen_standard = set()

            for name in raw.ch_names:
                # Isolate the primary electrode from bipolar names (e.g., FP1-F3 -> Fp1)
                clean = name.replace('EEG ', '').split('-')[0].strip().replace('FP', 'Fp').replace('Z', 'z')
                clean_low = clean.lower()

                if clean_low in std_names_low:
                    standard_label = std_names_low[clean_low]
                    # First time seeing this location: Make it the primary EEG channel
                    if standard_label not in seen_standard:
                        new_names[name] = standard_label
                        ch_types[standard_label] = 'eeg' 
                        seen_standard.add(standard_label)
                    else:
                        # Duplicate location: Rename and hide from spatial maps
                        unique_extra = f"Extra_{standard_label}_{name}"
                        new_names[name] = unique_extra
                        ch_types[unique_extra] = 'misc' 
                else:
                    new_names[name] = name
                    ch_types[name] = 'misc' # Hide non-standard channels from maps

            # Apply names and types
            raw.rename_channels(new_names)
            raw.set_channel_types(ch_types)
            
            # Apply montage (MNE will safely ignore 'misc' channels)
            raw.set_montage(montage_1020, on_missing='ignore')
            raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

            # --- 2. BIOMARKER CALCULATION (DAR) ---
            # 'picks="eeg"' ensures we only calculate math on the primary locations!
            psds = raw.compute_psd(fmin=1, fmax=40, picks='eeg', verbose=False)
            psd_vals, freqs = psds.get_data(return_freqs=True)
            
            delta_mask = (freqs >= 1) & (freqs <= 4)
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            
            delta_p = psd_vals[:, delta_mask].mean()
            alpha_p = psd_vals[:, alpha_mask].mean()
            
            dar = delta_p / (alpha_p + 1e-10) if alpha_p > 0 else 0.0
            if np.isnan(dar): dar = 0.0
            
            batch_dar_scores.append(float(dar))
            patient_records.append({
                "File": uploaded_file.name,
                "DAR Index": round(float(dar), 2),
                "Risk": "High" if dar > 2.3 else "Moderate" if dar > 1.4 else "Low"
            })

            # --- VISUALIZATIONS (First file only) ---
            if uploaded_file == uploaded_files[0]:
                with col1:
                    st.subheader(f"📊 Live Signal Analysis: {uploaded_file.name}")
                    # Allow viewing of ALL channels, even 'misc' ones
                    selected_ch = st.selectbox("Monitor Channel", raw.ch_names)
                    
                    if live_stream:
                        placeholder = st.empty()
                        sleep_t = {"Slow": 0.4, "Normal": 0.1, "Fast": 0.02}[stream_speed]
                        for i in range(0, 300, 10):
                            data, times = raw[raw.ch_names.index(selected_ch), i:i+150]
                            fig_l = px.line(x=times, y=data[0]*1e6, template="plotly_white", height=300)
                            fig_l.update_layout(yaxis_title="µV", xaxis_title="Time (s)")
                            placeholder.plotly_chart(fig_l, use_container_width=True)
                            time.sleep(sleep_t)
                    else:
                        data, times = raw[raw.ch_names.index(selected_ch), :1500]
                        st.plotly_chart(px.line(x=times, y=data[0]*1e6, template="plotly_white", height=300), use_container_width=True)

                    st.subheader("📍 Brain Activity Heatmap (XAI)")
                    
                    fig_map, ax_map = plt.subplots(figsize=(4, 4))
                    if raw.info['dig'] is not None and len(raw.info['dig']) > 0:
                        mne.viz.plot_topomap(np.mean(psd_vals, axis=1), raw.info, axes=ax_map, show=False, cmap='Reds', contours=0)
                        st.pyplot(fig_map)
                    else:
                        st.warning("Spatial metadata missing for heatmap visualization.")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    # --- 3. OVERALL CONCLUSION ---
    with col2:
        st.subheader("🏁 Overall Conclusion")
        if batch_dar_scores:
            avg_batch_dar = np.nanmean(batch_dar_scores)
            overall_risk_score = min(int((avg_batch_dar / 3.0) * 100), 100)
            
            st.metric("Batch Neuro-Anomaly Score", f"{overall_risk_score}%", f"{avg_batch_dar:.2f} Avg DAR")
            
            if overall_risk_score > 70:
                st.error("BATCH VERDICT: HIGH RISK PATTERNS DETECTED")
            else:
                st.success("BATCH VERDICT: NORMAL SPECTRAL DISTRIBUTION")
        else:
            st.warning("No valid DAR data found in batch.")

        st.divider()
        st.write("**Batch Summary Table**")
        st.dataframe(pd.DataFrame(patient_records), use_container_width=True, hide_index=True)

        if st.button("📄 Generate Batch PDF Report"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "MindGuard Batch Clinical Report", ln=True, align='C')
                pdf.set_font("Arial", size=10)
                pdf.ln(10)
                pdf.cell(0, 10, f"Total Samples Analyzed: {len(uploaded_files)}", ln=True)
                pdf.cell(0, 10, f"Average Batch DAR: {np.nanmean(batch_dar_scores):.2f}", ln=True)
                pdf.ln(5)
                for rec in patient_records:
                    pdf.cell(0, 8, f"- {rec['File']}: DAR {rec['DAR Index']} ({rec['Risk']} Risk)", ln=True)
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
                st.download_button("📥 Download PDF", data=pdf_bytes, file_name="MindGuard_Batch_Report.pdf")
            except Exception as pdf_err:
                st.error(f"PDF Error: {pdf_err}")

        st.divider()
        st.subheader("📍 Sensor Mapping")
        try:
            fig_s, ax_s = plt.subplots(figsize=(4, 4))
            raw.plot_sensors(show_names=True, axes=ax_s, show=False)
            ax_s.axis('off')
            st.pyplot(fig_s)
        except:
            st.info("Mapping unavailable.")

else:
    st.info("👋 Awaiting multiple file uploads for batch clinical analysis.")

st.divider()
st.caption("🧠Schizophrenia | Batch EEG")