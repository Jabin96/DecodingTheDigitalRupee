# Decoding the Digital Rupee: A Predictive Analysis of India's UPI Spending

## Team Members
- **Jabin Shalom S** (jabinshaloms@iisc.ac.in)
- **Suganya H** (suganyah@iisc.ac.in)

## Problem Statement
This project analyzes UPI transaction data to quantify the "Digital Divide" in India's digital economy. While UPI adoption is widespread, we aim to investigate whether economic power is concentrated among users with premium digital infrastructure (5G/iOS) compared to those with basic connectivity.

## Dataset
The project uses the **UPI Transactions 2024 Dataset** sourced from Kaggle.
- **Source**: Kaggle (UPI Transactions Generator)
- **Location**: `data/raw/upi_transactions_2024.csv`
- **Description**: Contains simulated UPI transaction data including transaction amount, device type, location, and merchant category.

## High-Level Approach and Methods
We employ a combination of data analysis, machine learning, and a custom "Realism Engine" to model and predict spending behavior.

1.  **Data Preprocessing**: Cleaning and transforming raw transaction data.
2.  **Machine Learning Models**: Training XGBoost and Random Forest regressors to predict transaction amounts.
3.  **Realism Engine**: A custom logic layer that adjusts predictions based on:
    - **Digital Privilege Multipliers**: Adjusting for device type (iOS vs. Android) and network speed (5G vs. 4G vs. 2G).
    - **Market Calibration**: Projecting 2024 data to 2025 trends using inflation and market growth factors.
    - **Sector-Tech Bias**: Simulating access barriers in high-value sectors for low-tech users.

## Summary of Results
1.  **The Digital Divide is Real**: Users with Modern Connectivity (4G & 5G) spend significantly more than those on basic networks.
2.  **Infrastructure = Economy**: 4G has become the economic baseline. To bridge the divide, upgrading the 'Low Tech' population to modern standards is essential.
3.  **Premium User Dominance**: High-value sectors are effectively "gated" by digital infrastructure, with premium users (iOS/5G) showing significantly higher spending power.

## Project Structure
- `data/`: Contains all data files.
    - `raw/`: Original dataset.
    - `processed/`: Processed data files (if any).
    - `external/`: External data sources.
- `app.py`: Main Streamlit application containing the UI and Realism Engine.
- `upispendingpredictiveanalysis_new.py`: Model training script (XGBoost/Random Forest).
- `setup_metadata.py`: Helper script to generate encoding metadata.
- `requirements.txt`: List of Python dependencies.
- `LICENSE`: MIT License file.

## Installation & Usage

### Prerequisites
- Python 3.8+
- Streamlit

### Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Metadata**:
    ```bash
    python setup_metadata.py
    ```
3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
