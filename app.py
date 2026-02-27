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

if uploaded_files:
    # --- Master Patient Selector ---
    st.sidebar.divider()
    file_options = ["🧠 Overall Batch Average"] + [f.name for f in uploaded_files]
    selected_file_name = st.sidebar.selectbox(
        "🔍 Select View", 
        file_options,
        help="View the group aggregate, or select a specific patient."
    )

    st.sidebar.divider()
    st.sidebar.header("Simulation Control")
    live_stream = st.sidebar.toggle("Simulate Live Analysis Stream")
    stream_speed = st.sidebar.select_slider("Stream Speed", options=["Slow", "Normal", "Fast"], value="Slow")

    col1, col2 = st.columns([2, 1])
    
    batch_dar_scores = []
    patient_records = []
    all_psd_data = [] 
    valid_info = None 
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # Load Data
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False) if uploaded_file.name.endswith('.edf') else mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)

            # --- ROBUST SPATIAL OVERLAP FIX ---
            montage_1020 = mne.channels.make_standard_montage('standard_1020')
            std_names_low = {n.lower(): n for n in montage_1020.ch_names}
            
            new_names = {}
            ch_types = {}
            seen_standard = set()

            for name in raw.ch_names:
                clean = name.replace('EEG ', '').split('-')[0].strip().replace('FP', 'Fp').replace('Z', 'z')
                clean_low = clean.lower()

                if clean_low in std_names_low:
                    standard_label = std_names_low[clean_low]
                    if standard_label not in seen_standard:
                        new_names[name] = standard_label
                        ch_types[standard_label] = 'eeg' 
                        seen_standard.add(standard_label)
                    else:
                        unique_extra = f"Extra_{standard_label}_{name}"
                        new_names[name] = unique_extra
                        ch_types[unique_extra] = 'misc' 
                else:
                    new_names[name] = name
                    ch_types[name] = 'misc' 

            raw.rename_channels(new_names)
            raw.set_channel_types(ch_types)
            raw.set_montage(montage_1020, on_missing='ignore')
            raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

            if valid_info is None and raw.info['dig'] is not None and len(raw.info['dig']) > 0:
                valid_info = raw.info

            # --- 2. BIOMARKER CALCULATION (DAR) ---
            psds = raw.compute_psd(fmin=1, fmax=40, picks='eeg', verbose=False)
            psd_vals, freqs = psds.get_data(return_freqs=True)
            
            all_psd_data.append(np.mean(psd_vals, axis=1))
            
            delta_mask = (freqs >= 1) & (freqs <= 4)
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            
            delta_p = psd_vals[:, delta_mask].mean()
            alpha_p = psd_vals[:, alpha_mask].mean()
            
            dar = delta_p / (alpha_p + 1e-10) if alpha_p > 0 else 0.0
            if np.isnan(dar): dar = 0.0
            
            risk_status = "High" if dar > 2.3 else "Moderate" if dar > 1.4 else "Low"
            batch_dar_scores.append(float(dar))

            # --- GENERATE PDF SIGNAL THUMBNAIL IN BACKGROUND ---
            # We take a quick 3-second snippet of the first available channel
            fig_thumb, ax_thumb = plt.subplots(figsize=(8, 1.5))
            thumb_data, thumb_times = raw[0, :int(raw.info['sfreq'] * 3)] 
            ax_thumb.plot(thumb_times, thumb_data[0] * 1e6, color='#1f77b4', linewidth=0.8)
            ax_thumb.axis('off') # Hide axes for a clean medical report look
            ax_thumb.set_title(f"Primary Trace ({raw.ch_names[0]})", fontsize=8, loc='left')
            
            # Save thumbnail to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_thumb:
                fig_thumb.savefig(tmp_thumb.name, format="png", bbox_inches="tight", dpi=100)
                thumb_path = tmp_thumb.name
            plt.close(fig_thumb) # Free memory!

            patient_records.append({
                "File": uploaded_file.name,
                "DAR Index": round(float(dar), 2),
                "Risk": risk_status,
                "Thumbnail": thumb_path
            })

            # --- INDIVIDUAL VISUALIZATIONS ---
            if uploaded_file.name == selected_file_name:
                with col1:
                    st.subheader(f"📊 Individual Analysis: {uploaded_file.name}")
                    st.metric(
                        label="Individual DAR Score", 
                        value=f"{dar:.2f}", 
                        delta=f"Risk Level: {risk_status}",
                        delta_color="inverse" if risk_status == "High" else "normal"
                    )

                    selected_ch = st.selectbox("Monitor Channel", raw.ch_names, key=f"ch_{uploaded_file.name}")
                    
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

                    st.subheader("Individual Brain Activity Heatmap")
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

    # --- AGGREGATE BATCH VISUALIZATION ---
    if selected_file_name == "🧠 Overall Batch Average":
        with col1:
            st.subheader("📊 Aggregate Batch Analysis")
            st.info("💡 You are viewing the aggregate statistical baseline of all uploaded patients. To view live signal streams, select an individual patient from the sidebar.")
            
            st.subheader("Aggregate Brain Activity Heatmap (Group Average)")
            if valid_info is not None and len(all_psd_data) > 0:
                avg_psd = np.mean(all_psd_data, axis=0)
                fig_map, ax_map = plt.subplots(figsize=(5, 5))
                mne.viz.plot_topomap(avg_psd, valid_info, axes=ax_map, show=False, cmap='Reds', contours=0)
                st.pyplot(fig_map)
            else:
                st.warning("Spatial metadata missing across the batch. Cannot generate average heatmap.")

    # --- 3. OVERALL CONCLUSION ---
    with col2:
        st.subheader("🏁 Overall Batch Conclusion")
        if batch_dar_scores:
            avg_batch_dar = np.nanmean(batch_dar_scores)
            overall_risk_score = min(int((avg_batch_dar / 3.0) * 100), 100)
            
            st.metric("Group Neuro-Anomaly Score", f"{overall_risk_score}%", f"{avg_batch_dar:.2f} Avg DAR")
            
            if overall_risk_score > 70:
                st.error("BATCH VERDICT: HIGH RISK PATTERNS DETECTED")
            else:
                st.success("BATCH VERDICT: NORMAL SPECTRAL DISTRIBUTION")
        else:
            st.warning("No valid DAR data found in batch.")

        st.divider()
        st.write("**Batch Summary Table**")
        st.dataframe(pd.DataFrame([{"File": r["File"], "DAR": r["DAR Index"], "Risk": r["Risk"]} for r in patient_records]), use_container_width=True, hide_index=True)

        # --- ADVANCED PDF GENERATION WITH INDIVIDUAL GRAPHS ---
        if st.button("📄 Generate Detailed Batch PDF Report"):
            try:
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18)
                pdf.cell(0, 10, "Schizophrenia: Comprehensive Clinical Report", ln=True, align='C')
                pdf.set_font("Arial", size=10)
                pdf.ln(5)
                
                # 1. Aggregate Heatmap Image to PDF
                if valid_info is not None and len(all_psd_data) > 0:
                    avg_psd = np.mean(all_psd_data, axis=0)
                    fig_pdf, ax_pdf = plt.subplots(figsize=(4, 4))
                    mne.viz.plot_topomap(avg_psd, valid_info, axes=ax_pdf, show=False, cmap='Reds', contours=0)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        fig_pdf.savefig(tmp_img.name, format="png", bbox_inches="tight", dpi=150)
                        tmp_img_path = tmp_img.name
                    plt.close(fig_pdf)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "1. Aggregate Group Heatmap", ln=True)
                    pdf.image(tmp_img_path, x=65, w=80) 
                    pdf.ln(5)
                    os.remove(tmp_img_path) # Clean up
                
                # 2. Add Overall Statistics
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "2. Group Statistical Baseline", ln=True)
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 8, f"Total Samples Analyzed: {len(uploaded_files)}", ln=True)
                pdf.cell(0, 8, f"Average Batch DAR: {np.nanmean(batch_dar_scores):.2f}", ln=True)
                pdf.cell(0, 8, f"Overall Batch Risk Level: {'HIGH' if overall_risk_score > 70 else 'NORMAL'}", ln=True)
                pdf.ln(10)
                
                # 3. Add Individual Breakdown with Graphs
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "3. Individual Patient Analysis & Trace Recordings", ln=True)
                pdf.ln(2)
                
                for rec in patient_records:
                    # Check if we need a new page so the graph doesn't get cut off
                    if pdf.get_y() > 240:
                        pdf.add_page()
                        
                    pdf.set_font("Arial", 'B', 10)
                    # Text Data
                    pdf.cell(0, 6, f"Patient: {rec['File']} | DAR: {rec['DAR Index']} | Risk: {rec['Risk']}", ln=True)
                    
                    # Graph Data
                    if os.path.exists(rec['Thumbnail']):
                        pdf.image(rec['Thumbnail'], x=15, w=160)
                        pdf.ln(2) # Spacing after image
                        os.remove(rec['Thumbnail']) # Clean up the temp image
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
                st.download_button("📥 Download Detailed PDF", data=pdf_bytes, file_name="Schizophrenia_Detailed_Report.pdf")
            except Exception as pdf_err:
                st.error(f"PDF Error: {pdf_err}")

else:
    st.info("👋 Awaiting multiple file uploads for batch clinical analysis.")

st.divider()
st.caption("Schizophrenia | Interactive Batch Engine")
