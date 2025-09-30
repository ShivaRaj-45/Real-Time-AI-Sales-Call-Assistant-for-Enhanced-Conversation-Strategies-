import streamlit as st
import pandas as pd
import datetime, time, json, os, re, queue, multiprocessing
from collections import deque
import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from groq import Groq
from deepmultilingualpunctuation import PunctuationModel
import gspread
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================
# === GOOGLE SHEETS CONFIGURATION ======================================================
# ======================================================================================

CRM_SHEET_NAME = "CRM Data"
PRODUCT_CATALOG_SHEET_NAME = "Product Catalog"
CRM_LOGS_SHEET_NAME = "CRM Logs"
CALL_SUMMARIES_SHEET_NAME = "Call Summaries"
CONVERSATION_LOGS_SHEET_NAME = "Conversation Logs" # <-- ADD THIS LINE
CREDENTIALS_FILE = "credentials.json"

def get_gspread_client():
    try:
        return gspread.service_account(filename=CREDENTIALS_FILE)
    except Exception as e:
        print(f"[ERROR] Google Sheets authentication failed: {e}")
        return None

def fetch_data_as_df(client, sheet_name, is_frontend=False):
    try:
        spreadsheet = client.open(sheet_name)
        worksheet = spreadsheet.sheet1
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        error_msg = f"Google Sheet named '{sheet_name}' was not found."
        if is_frontend: st.error(error_msg)
        else: print(f"[ERROR] {error_msg}")
        return None
    except Exception as e:
        error_msg = f"Failed to fetch data from '{sheet_name}'. Error: {e}"
        if is_frontend: st.error(error_msg)
        else: print(f"[ERROR] {error_msg}")
        return None
    
def export_summary_to_sheet(client, sheet_name, summary_text, customer_profile):
    """Appends a new row with the call summary to the specified sheet."""
    try:
        spreadsheet = client.open(sheet_name)
        worksheet = spreadsheet.sheet1
        
        # Create the data for the new row
        summary_id = f"SUM-{int(time.time())}"
        customer_id = customer_profile.get("CustomerID", "N/A")
        customer_name = customer_profile.get("Name", "N/A")
        call_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        new_row = [summary_id, customer_id, customer_name, call_date, summary_text]
        
        worksheet.append_row(new_row)
        print(f"Successfully exported summary {summary_id} to '{sheet_name}'.")
        return True
    except Exception as e:
        print(f"[EXPORT ERROR] Failed to export summary: {e}")
        return False
def export_conversation_to_sheet(client, sheet_name, conversation_log, customer_profile):
    """Appends all rows from a conversation log to the specified sheet."""
    if not conversation_log:
        print("[EXPORT ERROR] Conversation log is empty. Nothing to export.")
        return False
        
    try:
        spreadsheet = client.open(sheet_name)
        worksheet = spreadsheet.sheet1
        
        # Create a unique ID for this entire conversation
        conversation_id = f"CONVO-{int(time.time())}"
        customer_id = customer_profile.get("CustomerID", "N/A")
        customer_name = customer_profile.get("Name", "N/A")
        
        # Prepare all rows for batch insertion
        rows_to_append = []
        for entry in conversation_log:
            new_row = [
                conversation_id,
                customer_id,
                customer_name,
                entry.get("Time", ""),
                entry.get("Sentence", ""),
                entry.get("sentiment", ""),
                entry.get("inquiry_type", ""),
                entry.get("topic", "")
            ]
            rows_to_append.append(new_row)
            
        # Append all rows at once for efficiency
        worksheet.append_rows(rows_to_append)
        print(f"Successfully exported conversation {conversation_id} with {len(rows_to_append)} entries to '{sheet_name}'.")
        return True
        
    except Exception as e:
        print(f"[EXPORT ERROR] Failed to export conversation log: {e}")
        return False

# ======================================================================================
# === BACKEND LOGIC ====================================================================
# ======================================================================================

CONFIG = {
    "CHUNK": 1024, "FORMAT": pyaudio.paInt16, "CHANNELS": 1, "RATE": 16000,
    "WHISPER_MODEL": "small", "SILENCE_THRESHOLD": 350, "SILENCE_DURATION": 2,
    "GROQ_API_KEY": "GROQ_API_KEY"
}
AGENT_CONFIG = {
    "FAST_MODEL": "llama-3.1-8b-instant",
    "SMART_MODEL": "llama-3.3-70b-versatile",
    "CONTEXT_WINDOW": 3
}

def backend_process_runner(command_queue, results_queue, stop_event, shutdown_event):
    print("[BACKEND] Initializing...")
    try:
        punctuation_model = PunctuationModel()
        device, compute_type = (("cuda", "float16") if torch.cuda.is_available() else ("cpu", "int8"))
        print(f"[BACKEND] Using device: {device}")
        whisper_model = WhisperModel(CONFIG["WHISPER_MODEL"], device=device, compute_type=compute_type)
        groq_client = Groq(api_key=CONFIG["GROQ_API_KEY"])
        gspread_client = get_gspread_client()
        product_catalog = fetch_data_as_df(gspread_client, PRODUCT_CATALOG_SHEET_NAME)
        crm_logs = fetch_data_as_df(gspread_client, CRM_LOGS_SHEET_NAME)
        if product_catalog is None or crm_logs is None: raise Exception("Failed to fetch Product Catalog or CRM Logs.")
        results_queue.put({"status": "ready", "message": f"AI Engine ready on {device}."})
    except Exception as e:
        results_queue.put({"status": "error", "message": f"Fatal boot error: {e}"})
        return

    while not shutdown_event.is_set():
        try:
            command = command_queue.get(timeout=1)
            if command and command.get("action") == "START":
                customer_profile = command.get("profile")
                if customer_profile:
                    print(f"[BACKEND] Monitoring started for customer: {customer_profile.get('Name')}")
                    results_queue.put({"status": "monitoring_started", "message": f"Live call started with {customer_profile.get('Name')}..."})
                    live_transcribe(whisper_model, punctuation_model, groq_client, customer_profile, product_catalog, crm_logs, results_queue, stop_event, shutdown_event)
                    stop_event.clear()
                    results_queue.put({"status": "monitoring_stopped", "message": "Monitoring stopped."})
                    print("[BACKEND] Monitoring stopped.")
        except queue.Empty:
            continue
    print("[BACKEND] Shutdown signal received.")

def live_transcribe(whisper, punc, groq, customer_profile, product_catalog, crm_logs, results_queue, stop_event, shutdown_event):
    p = pyaudio.PyAudio()
    stream = p.open(format=CONFIG["FORMAT"], channels=CONFIG["CHANNELS"], rate=CONFIG["RATE"], input=True, frames_per_buffer=CONFIG["CHUNK"])
    audio_buffer, context_window, is_speaking, silence_start_time = [], deque(maxlen=AGENT_CONFIG["CONTEXT_WINDOW"]), False, None

    while not stop_event.is_set() and not shutdown_event.is_set():
        data = stream.read(CONFIG["CHUNK"], exception_on_overflow=False)
        rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(float) ** 2))
        if rms > CONFIG["SILENCE_THRESHOLD"]:
            is_speaking, silence_start_time = True, None
            audio_buffer.append(data)
        elif is_speaking:
            if silence_start_time is None: silence_start_time = time.time()
            if time.time() - silence_start_time > CONFIG["SILENCE_DURATION"]:
                if audio_buffer:
                    audio_np = (np.frombuffer(b"".join(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0)
                    segments, _ = whisper.transcribe(audio_np, beam_size=5, task="transcribe", language="en")
                    phrase = "".join(s.text for s in segments).strip()
                    if phrase:
                        punctuated_phrase = punc.restore_punctuation(phrase)
                        analysis = analyze_sentence_for_inquiry(punctuated_phrase, list(context_window), groq)
                        results_queue.put({"type": "data", "payload": analysis})
                        context_window.append(punctuated_phrase)
                        inquiry_type = analysis.get("inquiry_type", "statement")
                        if inquiry_type != "statement":
                            search_topic = analysis.get("topic", punctuated_phrase)
                            recommendations = recommend_product_hybrid(customer_profile, search_topic, product_catalog, crm_logs)
                            if recommendations:
                                results_queue.put({"type": "product_recommendation", "payload": recommendations})
                audio_buffer, is_speaking, silence_start_time = [], False, None
    stream.stop_stream()
    stream.close()
    p.terminate()

def analyze_sentence_for_inquiry(phrase, context, client):
    prompt = f"""
    You are an expert sales call analyst. Your job is to analyze the user's sentence based on three criteria: Sentiment, Inquiry Type, and Topic.
    **1. Sentiment Analysis:** Independently analyze the emotional tone of the sentence. Classify it as 'positive', 'neutral', or 'negative'.
    **2. Inquiry Type Classification:**
    - 'product_inquiry': User asks about or states a desire for a specific product/category.
    - 'scenario_inquiry': User describes a need or situation and wants a recommendation.
    - 'general_inquiry': User asks for general suggestions without specifics.
    - 'statement': General conversation that is NOT a product inquiry.
    **3. Topic Extraction:** Extract the main 'topic' of the sentence.
    **Examples:**
    Sentence: "I'm so excited to finally get a new camera!" -> {{"sentiment": "positive", "inquiry_type": "product_inquiry", "topic": "new camera"}}
    Sentence: "My old laptop is broken and I'm really frustrated." -> {{"sentiment": "negative", "inquiry_type": "statement", "topic": "broken laptop"}}
    **Respond ONLY with a valid JSON object** with keys: "sentiment", "inquiry_type", and "topic".
    **Analysis Task:**
    Context: "{" ".join(context)}"
    Sentence: "{phrase}"
    """
    try:
        response = client.chat.completions.create(model=AGENT_CONFIG["FAST_MODEL"], messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0)        
        # --- ADDED LOGIC TO HANDLE LISTS ---
        analysis_raw = json.loads(response.choices[0].message.content)
        if isinstance(analysis_raw, list) and analysis_raw:
            analysis = analysis_raw[0] # Take the first dictionary from the list
        elif isinstance(analysis_raw, dict):
            analysis = analysis_raw # It's already a dictionary
        else:
            # Fallback for unexpected formats
            analysis = {"sentiment": "error", "inquiry_type": "statement", "topic": "Invalid format from AI"}
        # --- END OF NEW LOGIC ---
            
    except Exception as e:
        analysis = {"sentiment": "error", "inquiry_type": "statement", "topic": str(e)}

    return {"Time": datetime.datetime.now().strftime("%H:%M:%S"), "Sentence": phrase, **analysis}

def recommend_product_hybrid(customer_profile, search_query, product_catalog, crm_logs, top_n=3, w_query=0.6, w_history=0.4):
    if product_catalog is None or product_catalog.empty: return []

    feature_cols = ['Name', 'Category', 'Description']
    available_cols = [col for col in feature_cols if col in product_catalog.columns]
    if 'Name' not in available_cols:
        print("[BACKEND ERROR] Product Catalog must have a 'Name' column.")
        return []

    try:
        product_catalog["combined_features"] = product_catalog[available_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(product_catalog["combined_features"])
        
        query_vector = vectorizer.transform([search_query])
        query_similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        is_query_match_poor = np.max(query_similarity) < 0.1

        history_similarity = np.zeros(len(product_catalog))
        customer_id = customer_profile.get('CustomerID')
        past_purchases_ids = []
        if customer_id and 'CustomerID' in crm_logs.columns:
            customer_logs_df = crm_logs[crm_logs['CustomerID'] == customer_id]
            past_purchases_ids = customer_logs_df['ProductID_Purchased'].unique().tolist()

            if past_purchases_ids:
                item_item_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
                for product_id in past_purchases_ids:
                    try:
                        idx = product_catalog.index[product_catalog['ProductID'] == product_id].tolist()[0]
                        history_similarity += item_item_similarity[idx]
                    except IndexError:
                        continue
        
        reason_template = ""
        if is_query_match_poor and np.any(history_similarity):
            final_scores = history_similarity
            reason_template = "While we couldn't find products for '{query}', you might like these based on your purchase history:"
        else:
            if np.any(history_similarity):
                history_similarity = (history_similarity - history_similarity.min()) / (history_similarity.max() - history_similarity.min())
            final_scores = (w_query * query_similarity) + (w_history * history_similarity)
            reason_template = "Based on your interest in '{query}' and your purchase history:"

        boost_score = 2.0 
        simple_search_term = search_query.split()[-1]
        for i, name in enumerate(product_catalog['Name']):
            if simple_search_term.lower() in name.lower():
                final_scores[i] += boost_score

        top_indices = final_scores.argsort()[-top_n*2:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= top_n: break
            product = product_catalog.iloc[idx]
            if 'ProductID' in product and product['ProductID'] not in past_purchases_ids:
                reason = reason_template.format(query=search_query)
                recommendations.append({"Name": product["Name"], "Category": product.get("Category", "N/A"), "Reason": reason})
        return recommendations
    except Exception as e:
        print(f"[BACKEND ERROR] An error occurred during recommendation: {e}")
        return []

# <--- NEW FUNCTION FOR POST-CALL SUMMARY --->
def generate_summary(conversation_log, client):
    if not conversation_log:
        return "No conversation was recorded."

    # Format the log for the AI
    formatted_log = "\n".join([f"- {item['Sentence']}" for item in conversation_log])

    prompt = f"""
    You are an expert sales manager reviewing a call transcript.
    Your task is to provide a concise, structured summary of the call.

    **Call Transcript:**
    {formatted_log}

    **Your Summary (in markdown format):**
    Provide the following sections:
    - **Overall Mood:** A one-sentence summary of the call's sentiment.
    - **Customer's Main Goal:** What was the customer trying to achieve or inquire about?
    - **Key Topics Discussed:** A bulleted list of the main topics.
    - **Action Items:** Any follow-up actions required. If none, state "No action items."
    """
    try:
        response = client.chat.completions.create(
            model=AGENT_CONFIG["SMART_MODEL"], # Use the smart model for a high-quality summary
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"[SUMMARY ERROR] LLM failed to generate summary: {e}")
        return f"Error generating summary: {e}"

# ======================================================================================
# === MAIN FRONTEND APPLICATION ========================================================
# ======================================================================================

def main_frontend():
    st.set_page_config(page_title="AI Sales Call Assistant", layout="wide")

    if 'app_state' not in st.session_state:
        st.session_state.app_state = "BOOTING"
        st.session_state.backend_status = "Engine is booting..."
        st.session_state.customer_profile = None
        st.session_state.recommendations = []
        st.session_state.rows = []
        st.session_state.crm_data = None
        st.session_state.call_summary = None # <-- Initialize summary state
        st.session_state.summary_exported = False # You already have this
        st.session_state.log_exported = False 

        st.session_state.manager = multiprocessing.Manager()
        st.session_state.command_queue = st.session_state.manager.Queue()
        st.session_state.results_queue = st.session_state.manager.Queue()
        st.session_state.shutdown_event = st.session_state.manager.Event()
        st.session_state.stop_event = st.session_state.manager.Event() 
        
        st.session_state.backend_process = multiprocessing.Process(target=backend_process_runner, args=(st.session_state.command_queue, st.session_state.results_queue, st.session_state.stop_event, st.session_state.shutdown_event))
        st.session_state.backend_process.start()

    st.title("🚀 AI Sales Call Assistant")
    st.info(f"**Status:** {st.session_state.backend_status}")
    
    col_main, col_sidebar = st.columns([2, 1])

    with col_sidebar:
        st.header("👤 Customer Zone")
        with st.container(border=True):
            # --- Load all data into session state if not already loaded ---
            if 'data_loaded' not in st.session_state:
                client = get_gspread_client()
                if client:
                    with st.spinner("Connecting to Google Sheets and loading data..."):
                        st.session_state.crm_data = fetch_data_as_df(client, CRM_SHEET_NAME, is_frontend=True)
                        st.session_state.product_catalog = fetch_data_as_df(client, PRODUCT_CATALOG_SHEET_NAME, is_frontend=True)
                        st.session_state.crm_logs = fetch_data_as_df(client, CRM_LOGS_SHEET_NAME, is_frontend=True)
                        st.session_state.data_loaded = True # Mark as loaded
                        st.rerun()
                else:
                    st.error("Could not connect to Google Sheets.")
            if st.session_state.crm_data is None:
                st.warning("Waiting for CRM data to load...")
            elif 'Email' not in st.session_state.crm_data.columns:
                st.error(f"**CRM Error:** 'Email' column not found.")
            else:
                email = st.text_input("Enter customer email to load profile:", key="email_input")
                if st.button("Find & Load Customer"):
                    customer_df = st.session_state.crm_data[st.session_state.crm_data['Email'].str.lower() == email.lower().strip()]
                    if not customer_df.empty:
                        st.session_state.customer_profile = customer_df.iloc[0].to_dict()
                    else:
                        st.session_state.customer_profile = None
            
            if st.session_state.customer_profile:
                st.success(f"**Welcome, {st.session_state.customer_profile['Name']}!**")
                
                profile = st.session_state.customer_profile
                profile_df = pd.DataFrame(profile.items(), columns=['Field', 'Value'])
                st.table(profile_df)

                # --- THIS IS THE LOGIC THAT DISPLAYS THE PURCHASE HISTORY ---
                st.subheader("Purchase History")
                
                # Get the required data from session state
                crm_logs = st.session_state.crm_logs
                product_catalog = st.session_state.product_catalog
                customer_id = profile.get('CustomerID')

                # Find purchase logs for the current customer
                customer_purchases = crm_logs[crm_logs['CustomerID'] == customer_id]
                
                if customer_purchases.empty:
                    st.info("No purchase history found.")
                else:
                    # Get the list of Product IDs they purchased
                    purchased_ids = customer_purchases['ProductID_Purchased'].tolist()
                    # Filter the main product catalog to get the names of those products
                    purchased_products = product_catalog[product_catalog['ProductID'].isin(purchased_ids)]
                    
                    # Display the names of purchased products as a list
                    for product_name in purchased_products['Name']:
                        st.markdown(f"- {product_name}")
                # --- END OF THE LOGIC ---

            else:
                st.warning("No customer loaded.")
        if st.session_state.call_summary:
            with st.expander("📝 View Post-Call Summary", expanded=True):
                st.markdown(st.session_state.call_summary)
                
                # --- ADD THIS NEW BUTTON LOGIC ---
                if 'summary_exported' not in st.session_state:
                    st.session_state.summary_exported = False

                if st.button("Export Summary to Sheet", disabled=st.session_state.summary_exported):
                    # Re-initialize the client if it's not in the session state
                    if 'gspread_client' not in st.session_state or st.session_state.gspread_client is None:
                        st.session_state.gspread_client = get_gspread_client()

                    if st.session_state.gspread_client:
                        with st.spinner("Exporting..."):
                            success = export_summary_to_sheet(
                                st.session_state.gspread_client,
                                CALL_SUMMARIES_SHEET_NAME,
                                st.session_state.call_summary,
                                st.session_state.customer_profile
                            )
                        if success:
                            st.success("Summary exported successfully!")
                            st.session_state.summary_exported = True
                            st.rerun()
                        else:
                            st.error("Failed to export summary. Check console for details.")
                    else:
                        st.error("Cannot export: Google Sheets client not available.")
    with col_main:
        st.header("📞 Live Call")
        can_start_monitoring = (st.session_state.customer_profile is not None) and (st.session_state.app_state == "READY")
        is_monitoring = st.session_state.app_state == "MONITORING"
        
        c1, c2 = st.columns(2)
        if c1.button("▶️ Start Call Monitoring", disabled=not can_start_monitoring or is_monitoring):
            st.session_state.stop_event.clear()
            st.session_state.rows = [] 
            st.session_state.recommendations = []
            st.session_state.call_summary = None
            st.session_state.summary_exported = False # <-- ADD THIS LINE
            st.session_state.log_exported = False # <-- ADD THIS LINE
            command = {"action": "START", "profile": st.session_state.customer_profile}
            st.session_state.command_queue.put(command)
            st.rerun()
        
        # <--- UPDATED LOGIC FOR STOP BUTTON --->
        if c2.button("⏹️ Stop Call Monitoring", disabled=not is_monitoring):
            st.session_state.stop_event.set()
            with st.spinner("AI is generating the post-call summary..."):
                # Give the backend a moment to stop
                time.sleep(1) 
                # Generate summary in the frontend for immediate feedback
                client = Groq(api_key=CONFIG["GROQ_API_KEY"])
                summary = generate_summary(st.session_state.rows, client)
                st.session_state.call_summary = summary
                # Clear live data for the next call
                st.session_state.rows = []
                st.session_state.recommendations = []
            st.rerun()

        # <--- NEW UI SECTION FOR SUMMARY --->
        

# --- ADD THIS ENTIRE BLOCK FOR THE CONVERSATION TABLE AND BUTTON ---
        st.subheader("Conversation Insights")

        # First, display the table if there is data, otherwise show a message
        if st.session_state.rows:
            st.dataframe(st.session_state.rows, height=300, use_container_width=True)
        else:
            st.info("The live conversation transcript will appear here once the call starts.")

        # Second, add the export button below the table
        if st.button("Export Conversation Log",
                      disabled=(not st.session_state.rows) or st.session_state.get('log_exported', False)):
            
            # Check for gspread client
            if 'gspread_client' not in st.session_state or st.session_state.gspread_client is None:
                st.session_state.gspread_client = get_gspread_client()
            
            # If client is available, proceed with export
            if st.session_state.gspread_client:
                with st.spinner("Exporting conversation log..."):
                    success = export_conversation_to_sheet(
                        st.session_state.gspread_client,
                        CONVERSATION_LOGS_SHEET_NAME,
                        st.session_state.rows,
                        st.session_state.customer_profile
                    )
                if success:
                    st.success("Conversation log exported successfully!")
                    st.session_state.log_exported = True
                    st.rerun()
                else:
                    st.error("Failed to export log. Check console for details.")
            else:
                st.error("Cannot export: Google Sheets client not available.")
        
        st.subheader("💡 Product Recommendations")
        with st.container(height=300, border=True):
            if not st.session_state.recommendations: st.info("Waiting for customer inquiry...")
            else:
                for rec in st.session_state.recommendations:
                    st.markdown(f"**{rec['Name']}** ({rec.get('Category', 'N/A')})")
                    st.caption(f"Reason: {rec['Reason']}")
                    st.divider()

    if st.session_state.app_state != "TERMINATED":
        try:
            result = st.session_state.results_queue.get_nowait()
            if status := result.get("status"):
                st.session_state.backend_status = result.get("message", "Status updated.")
                if status in ["ready", "monitoring_stopped"]: st.session_state.app_state = "READY"
                elif status == "monitoring_started": st.session_state.app_state = "MONITORING"
                elif status == "error": st.session_state.app_state = "ERROR"
                st.rerun()
            if payload_type := result.get("type"):
                if payload_type == "data": st.session_state.rows.append(result["payload"])
                elif payload_type == "product_recommendation": st.session_state.recommendations = result["payload"]
                st.rerun()
        except queue.Empty:
            if st.session_state.app_state in ["BOOTING", "MONITORING"]: time.sleep(0.5); st.rerun()

if __name__ == "__main__":
    main_frontend()
