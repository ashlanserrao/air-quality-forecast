# Air Quality Forecasting Using LSTM Models: A Study in Delhi

Forecasting **daily PM₂.₅** in **Delhi** using a univariate **LSTM** model trained on **open data (OpenAQ v3)**.  
This repo includes the full, reproducible pipeline: **data collection → cleaning → EDA → modeling → evaluation → Streamlit app**.

> **Final model (daily, univariate LSTM)**
> - **RMSE:** ~31.09 µg/m³  
> - **MAE:** ~19.24 µg/m³  
> - Lookback window: **30 days** → next-day prediction

---

##  Project Highlights

- **Open data**: Fetched from **OpenAQ API v3** (Location: *New Delhi*, ID **8118**, Sensor for PM₂.₅).
- **End-to-end pipeline**: API data → parquet/CSV → cleaning → daily aggregation → LSTM training.
- **Univariate**: Uses only PM₂.₅ history (robust when met data is missing).
- **Deployment**: Interactive **Streamlit** app for 7–30 day forecasts.
- **Reproducible**: Scripts + notebooks + pinned requirements.

---

##  Installation

```bash
# Clone this repository
git clone https://github.com/ashlanserrao/air-quality-forecast.git
cd air-quality-forecast

# Create a virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run this command
streamlit run src/app.py
```
---

##  Results Summary
> - **RMSE:** 31.09 µg/m³  
> - **MAE:** 19.24 µg/m³  

- The model effectively captures Delhi’s daily PM₂.₅ trends and seasonality.
- It performs comparably to hybrid deep learning models while remaining lightweight and interpretable.
- The Streamlit app demonstrates practical usability for policymakers and researchers.

---

##  Model Overview

- **Model Type**: Long Short-Term Memory (LSTM) Neural Network
- **Input**: Previous 30 days of PM₂.₅ concentrations
- **Output**: Next day’s PM₂.₅ forecast
- **Optimizer**: Adam (lr = 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: RMSE, MAE
- **Train/Test Split**: 80% / 20% (chronological)

---

##  Future Work

- Add meteorological features (temperature, RH, wind) → multivariate LSTM/CNN–LSTM

- Multi-station, spatial modeling for Delhi

- Attention/Transformer variants

- Automated daily pipeline + hosted dashboard

---