# File: crm_connector.py
import pandas as pd

CRM_FILE = "crm_data.csv"

def get_customer_profile(customer_id):
    """
    Return a customer profile dict from CRM CSV.
    """
    try:
        df = pd.read_csv(CRM_FILE)
        row = df[df['CustomerID'] == customer_id]
        if row.empty:
            return None
        # Convert Interests column from string to list
        interests = row.iloc[0].get("Interests", "")
        if isinstance(interests, str):
            interests = [i.strip() for i in interests.split(",") if i.strip()]
        profile = {
            "CustomerID": customer_id,
            "Name": row.iloc[0].get("Name", ""),
            "Interests": interests,
            "Budget": row.iloc[0].get("Budget", ""),
            "LastPurchase": row.iloc[0].get("LastPurchase", "")
        }
        return profile
    except Exception as e:
        print(f"[crm_connector] Error: {e}")
        return None
