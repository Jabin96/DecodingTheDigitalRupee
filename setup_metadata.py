import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def create_metadata():
    print("Loading dataset...")
    try:
        df = pd.read_csv('data/raw/upi_transactions_2024.csv')
    except FileNotFoundError:
        print("Error: data/raw/upi_transactions_2024.csv not found.")
        return

    print("Extracting metadata...")
    metadata = {}
    encoders = {}
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"Processing {col}...")
        # Get unique values
        unique_vals = df[col].dropna().unique().astype(str)
        metadata[col] = unique_vals.tolist()
        
        # Fit encoder
        le = LabelEncoder()
        le.fit(unique_vals)
        encoders[col] = le
        
    print("Saving metadata to app_metadata.pkl...")
    joblib.dump({'options': metadata, 'encoders': encoders}, 'app_metadata.pkl')
    print("Done!")

if __name__ == "__main__":
    create_metadata()
