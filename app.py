# File: app.py

import streamlit as st
import pandas as pd
import datetime
import time
import json
import os
import io

st.set_page_config(page_title="AI Sales Call Assistant", layout="wide")

# --- CONFIGURATION ---
COMMUNICATION_FILE = "communication.jsonl"

# --- HEADER ---
st.title("🚀 AI Sales Call Assistant")
st.markdown("Run the `main.py` script in a separate terminal, then click 'Start Monitoring' here.")

# --- SESSION STATE ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "rows" not in st.session_state:
    st.session_state.rows = []
if "processed_lines" not in st.session_state:
    st.session_state.processed_lines = 0

# --- HELPER FUNCTIONS ---
def add_log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{ts}] {msg}")

def read_new_data():
    """Reads new lines from the communication file and updates session state."""
    new_data_found = False
    if not os.path.exists(COMMUNICATION_FILE):
        return new_data_found

    try:
        with open(COMMUNICATION_FILE, "r") as f:
            all_lines = f.readlines()
        
        if len(all_lines) > st.session_state.processed_lines:
            new_data_found = True
            new_lines = all_lines[st.session_state.processed_lines:]
            
            for line in new_lines:
                try:
                    data_row = json.loads(line)
                    st.session_state.rows.append(data_row)
                    add_log(f"Analyzed: \"{data_row['Sentence']}\" -> {data_row['Sentiment']}")
                except json.JSONDecodeError:
                    add_log(f"Warning: Could not decode a line: {line.strip()}")
            
            st.session_state.processed_lines = len(all_lines)

    except Exception as e:
        add_log(f"Error reading communication file: {e}")
        
    return new_data_found

# --- BUTTONS AND CONTROLS ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ Start Monitoring", disabled=st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = True
        add_log("Monitoring started. Waiting for data from backend...")
        st.rerun()

with col2:
    if st.button("⏹ Stop Monitoring", disabled=not st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = False
        add_log("Monitoring stopped.")
        st.rerun()

with col3:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.logs, st.session_state.rows = [], []
        st.session_state.processed_lines = 0
        if os.path.exists(COMMUNICATION_FILE):
            os.remove(COMMUNICATION_FILE)
        add_log("Dashboard cleared.")
        st.rerun()

# Display current status
status_placeholder = st.empty()
if st.session_state.is_running:
    status_placeholder.success("🟢 Monitoring Live", icon="📡")
else:
    status_placeholder.warning("🟠 Monitoring Stopped", icon="🛑")

# --- DASHBOARD LAYOUT ---
df = pd.DataFrame(st.session_state.rows) if st.session_state.rows else pd.DataFrame(
    columns=["Time", "Sentence", "Intent", "Sarcasm", "Sentiment", "Reasoning"]
)

left, right = st.columns([1, 2])

with left:
    st.subheader("Live Transcript Log")
    log_box = st.container(height=300, border=True)
    for log in st.session_state.logs:
        log_box.write(log)

    st.subheader("Sentiment Trend")
    if not df.empty and "Sentiment" in df.columns:
        sentiment_map = {"Positive": 1, "Negative": -1, "Neutral": 0, "Error": 0}
        df_chart = pd.DataFrame({
            "time": range(len(df)),
            "sentiment": df["Sentiment"].map(sentiment_map).fillna(0)
        })
        st.line_chart(df_chart, x="time", y="sentiment")

with right:
    st.subheader("Conversation Insights")
    st.dataframe(df, use_container_width=True, height=500)

# --- EXPORT HANDLERS (Excel Only) ---
st.markdown("---")
st.subheader("Export Call Summary")

if not df.empty:
    towrite_excel = io.BytesIO()
    df.to_excel(towrite_excel, index=False, sheet_name="Summary")
    towrite_excel.seek(0)
    st.download_button(
        label="📊 Export to Excel",
        data=towrite_excel,
        file_name="call_summary.xlsx",
        mime="application/vnd.ms-excel",
        use_container_width=True
    )
else:
    st.info("No data to export yet. Start monitoring and speak into your microphone.")

# --- AUTO-REFRESH LOOP ---
if st.session_state.is_running:
    if read_new_data():
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()