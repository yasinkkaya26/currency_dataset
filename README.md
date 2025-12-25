# Currency Time-Series Forecasting (Deep Learning)

This project performs **next-day Bitcoin closing price forecasting** as a **time-series regression** task.  
We compare multiple deep learning models under the same data split and evaluation metrics.

## Problem Definition
- **Target:** predict `close_USD(t+1)`
- **Input:** past **90 days** window (90 → 1 setup) with multivariate features
- **Type:** regression
- **Metrics:** MSE, RMSE, MAE, MAPE, R²

## Dataset
- File: `dc_extended.csv`
- Daily OHLCV (USD) + engineered indicators (e.g., returns, EMA, RSI)
- **Chronological split:** Train / Validation / Test (prevents leakage)
- Scaling is **fit on train only** and applied to val/test.

## Models
- **ResidualMLP:** strong baseline, stable generalization
- **LSTM:** temporal dependencies
- **Transformer:** self-attention encoder
- (Optional) **Hybrid** model (if enabled)

Model definitions are in:
- `models/mlp.py`
- `models/lstm.py`
- `models/transformer.py`

## Project Structure
CURRENCY_DATASET/
├─ main.py
├─ dc_extended.csv
├─ final_results.csv
├─ results_table.csv
├─ requirements.txt
├─ train.ipynb
├─ extended_dataset.py
├─ models/
│ ├─ init.py
│ ├─ mlp.py
│ ├─ lstm.py
│ ├─ transformer.py
│ ├─ mlp_best.pth
│ ├─ lstm_best.pth
│ └─ transformer_best.pth
├─ utils/
│ ├─ init.py
│ ├─ data.py
│ ├─ train.py
│ └─ plots.py
├─ plots/ # generated figures (all plots)
└─ checkpoints/ # saved model weights (.pt)
├─ residualmlp_phase1.pt
├─ residualmlp_final.pt
├─ lstm_phase1.pt
├─ lstm_final.pt
├─ transformer_phase1.pt
├─ transformer_final.pt
└─ ...


## Installation
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

Install dependencies
pip install -r requirements.txt
```


## Outputs

After running `python main.py`, the project produces the following artifacts:

### 1) Metrics & Tables
- **Final comparison table:** `final_results.csv`  
  Includes **RMSE, MAE, MAPE, R²** for each model.
- **Additional table (if generated):** `results_table.csv`

### 2) Plots (Visualization)
All figures are saved under `plots/`. Typical plots include:
- Training vs Validation Loss (per model)
- True vs Predicted (line plot, per model)
- True vs Predicted (scatter plot, per model)
- Error Distribution (absolute + percentage)
- All-model comparison (RMSE / MAE / MAPE / R²)

### 3) Saved Model Weights (Checkpoints)
Model checkpoints are saved under `checkpoints/`:
- `residualmlp_phase1.pt`, `residualmlp_final.pt`
- `lstm_phase1.pt`, `lstm_final.pt`
- `transformer_phase1.pt`, `transformer_final.pt`
- *(Optional)* `hybrid.pt` (if enabled)

> Tip: `.pt` files store model weights and can be loaded later for inference without retraining.

## Future Work

This project can be extended in several directions:

### 1) Better Targets (Non-Stationary Series)
- Predict **log-returns** instead of raw prices to reduce non-stationarity.
- Predict **price change (delta)** and reconstruct prices if needed.

### 2) Multi-step Forecasting
- Extend from one-step forecasting to **multi-step** horizons (e.g., 7-day ahead).
- Compare **direct multi-step** vs **recursive** forecasting strategies.

### 3) Exogenous Signals
- Add external features such as **macro indicators**, **sentiment**, or **on-chain** signals.
- Evaluate improvements during **high-volatility** or **regime-shift** periods.

### 4) Transformer Improvements
- Run broader **hyperparameter tuning** (layers, heads, `d_model`, dropout).
- Use **more data** and **longer context windows** to better exploit attention.












