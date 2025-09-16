# File: app.py

import streamlit as st
import pandas as pd
import datetime, time, json, os, io
import gspread
from gspread_dataframe import set_with_dataframe

st.set_page_config(page_title="AI Sales Call Assistant", layout="wide")
COMMUNICATION_FILE = "communication.jsonl"
GOOGLE_SHEET_NAME = "Sales Call Sentiment Analysis" 

st.title(" AI Sales Call Assistant")
st.markdown("Run `main.py` in a separate terminal, then click 'Start Monitoring'.")

if "is_running" not in st.session_state: st.session_state.is_running = False
if "logs" not in st.session_state: st.session_state.logs = []
if "rows" not in st.session_state: st.session_state.rows = []
if "processed_lines" not in st.session_state: st.session_state.processed_lines = 0
if "last_selection" not in st.session_state: st.session_state.last_selection = None

def add_log(msg):
    st.session_state.logs.insert(0, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

def read_new_data():
    if not os.path.exists(COMMUNICATION_FILE): return False
    try:
        with open(COMMUNICATION_FILE, "r") as f: all_lines = f.readlines()
        if len(all_lines) > st.session_state.processed_lines:
            for line in all_lines[st.session_state.processed_lines:]:
                data_row = json.loads(line)
                st.session_state.rows.append(data_row)
                add_log(f"Analyzed: \"{data_row['Sentence']}\" -> {data_row['Sentiment']}")
            st.session_state.processed_lines = len(all_lines)
            return True
    except Exception as e:
        add_log(f"Error reading file: {e}")
    return False

def highlight_sentiment(row):
    color = ''
    if row.get('Sentiment') == 'Positive': color = '#1E4620'
    elif row.get('Sentiment') == 'Negative': color = '#55262B'
    return [f'background-color: {color}'] * len(row)

def export_to_google_sheets(df_to_export):
    try:
        gc = gspread.service_account(filename='credentials.json')
        spreadsheet = gc.open(GOOGLE_SHEET_NAME)
        worksheet = spreadsheet.sheet1
        worksheet.clear()
        set_with_dataframe(worksheet, df_to_export)
        return f" Exported to '{GOOGLE_SHEET_NAME}'!"
    except Exception as e:
        return f" Error: {e}"

c1, c2, c3 = st.columns(3)
with c1:
    if st.button(" Start Monitoring", disabled=st.session_state.is_running, use_container_width=True):
        st.session_state.rows, st.session_state.logs, st.session_state.processed_lines = [], [], 0
        if os.path.exists(COMMUNICATION_FILE): os.remove(COMMUNICATION_FILE)
        st.session_state.is_running = True
        add_log("New call started. Monitoring...")
        st.rerun()
with c2:
    if st.button(" Stop Monitoring", disabled=not st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = False
        add_log("Monitoring stopped.")
        st.rerun()
with c3:
    if st.button(" Clear", use_container_width=True):
        st.session_state.rows, st.session_state.logs, st.session_state.processed_lines = [], [], 0
        if os.path.exists(COMMUNICATION_FILE): os.remove(COMMUNICATION_FILE)
        add_log("Dashboard cleared.")
        st.rerun()

status_placeholder = st.empty()
if st.session_state.is_running: status_placeholder.success(" Monitoring Live", icon="")
else: status_placeholder.warning(" Monitoring Stopped", icon="")

df = pd.DataFrame(st.session_state.rows)

# --- KEY CHANGE: More precise fix for data types to restore colors ---
if not df.empty:
    # Only convert the 'Reasoning' column to string, as it's the only one
    # that might have inconsistent data types. This leaves other columns alone.
    if 'Reasoning' in df.columns:
        df['Reasoning'] = df['Reasoning'].astype(str)

display_df = df.drop(columns=['Sarcasm'], errors='ignore') if not df.empty else pd.DataFrame(columns=["Time", "Sentence", "Intent", "Sentiment", "Reasoning"])

left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Live Transcript Log")
    log_box = st.container(height=300, border=True)
    for log in st.session_state.logs: log_box.write(log)
with right_col:
    st.subheader("Current Insight")
    insight_box = st.container(height=300, border=True)
    selected_row_index = st.session_state.last_selection["selection"]["rows"][0] if st.session_state.last_selection and st.session_state.last_selection["selection"]["rows"] else -1
    insight_data = df.iloc[selected_row_index] if selected_row_index != -1 else df.iloc[-1] if not df.empty else None
    if insight_data is not None:
        color = "green" if insight_data.get('Sentiment') == 'Positive' else "red" if insight_data.get('Sentiment') == 'Negative' else "orange"
        insight_box.markdown(f"**Intent:** {insight_data.get('Intent')}\n\n**Sentiment:** :{color}[{insight_data.get('Sentiment')}]\n\n**Full Reasoning:**\n{insight_data.get('Reasoning')}")
    else: insight_box.info("Waiting for insights...")

st.subheader("Conversation Insights")
st.markdown("ℹ *Click a row to view its full details in the 'Current Insight' box above.*")
if not display_df.empty:
    st.dataframe(display_df.style.apply(highlight_sentiment, axis=1), use_container_width=True, on_select="rerun", selection_mode="single-row", key="last_selection")

if not df.empty:
    st.markdown("---")
    st.subheader("Export Call Summary")
    export_c1, export_c2 = st.columns(2)
    with export_c1:
        towrite_excel = io.BytesIO()
        df.to_excel(towrite_excel, index=False, sheet_name="Summary")
        towrite_excel.seek(0)
        st.download_button(label=" Export to Excel", data=towrite_excel, file_name="call_summary.xlsx", use_container_width=True)
    with export_c2:
        if st.button(" Export to Google Sheets", use_container_width=True):
            with st.spinner("Exporting..."): message = export_to_google_sheets(df)
            if "" in message: st.success(message)
            else: st.error(message)

if st.session_state.is_running:
    if read_new_data(): st.rerun()
    else: time.sleep(1); st.rerun()
