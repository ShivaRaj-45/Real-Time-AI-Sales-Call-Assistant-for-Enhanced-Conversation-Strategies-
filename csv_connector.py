# File: csv_connector.py
import pandas as pd
import io

CRM_FILE = "crm_data.csv"

def get_crm_csv_bytes():
    """
    Return the CRM CSV as bytes for Streamlit download button.
    """
    try:
        df = pd.read_csv(CRM_FILE)
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        return csv_bytes
    except Exception as e:
        print(f"[csv_connector] Error: {e}")
        return None
