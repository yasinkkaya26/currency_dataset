import os
import torch
import pandas as pd
import numpy as np

from utils.data import load_time_series_dataloaders
from utils.train import train_model, evaluate, regression_metrics, retrain_on_train_val

from models.mlp import ImprovedMLP, ResidualMLP
from models.lstm import ImprovedLSTM
from models.transformer import TimeSeriesTransformer
from utils.plots import create_all_plots

CSV_PATH = "dc_extended.csv"
WINDOW_SIZE = 90
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print("\n" + "="*70)
print("TIME SERIES FORECASTING - OPTIMIZED TRAINING")
print("="*70)


print("\n=== Loading Data ===")
data = load_time_series_dataloaders(
    csv_path=CSV_PATH,
    target_col="close_USD",
    window_size=WINDOW_SIZE,
    batch_size=BATCH_SIZE,
)

train_loader = data["train_loader"]
val_loader = data["val_loader"]
test_loader = data["test_loader"]
target_scaler = data["target_scaler"]
seq_len = data["seq_len"]
num_features = data["num_features"]

print(f"Sequence length: {seq_len}")
print(f"Number of features: {num_features}")
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")


model_configs = {
    "ResidualMLP": {
        "class": ResidualMLP,
        "kwargs": {
            "seq_len": seq_len,
            "num_features": num_features,
            "dropout": 0.25
        },
        "lr": 8e-4,
        "weight_decay": 5e-6,
    },

    "LSTM": {
        "class": ImprovedLSTM,
        "kwargs": {
            "num_features": num_features,
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.25
        },
        "lr": 8e-4,
        "weight_decay": 5e-6,
    },

    "Transformer": {
        "class": TimeSeriesTransformer,
        "kwargs": {
            "num_features": num_features,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.25,
        },
        "lr": 8e-4,
        "weight_decay": 5e-6,
    },
}


all_results = []
all_test_predictions = {}

for model_name, config in model_configs.items():
    print("\n" + "="*70)
    print(f"=== {model_name.upper()} ===")
    print("="*70)

    print(f"\n[Phase 1] Training with separate train and validation sets...")

    model = config["class"](**config["kwargs"])

    model_trained, train_losses, val_losses, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=PATIENCE,
        name=f"{model_name.lower()}_phase1",
        device=DEVICE,
        scheduler_type="plateau",
        return_best_epoch=True,
    )

    print(f"Best epoch found: {best_epoch}")

    retrain_epochs = int(best_epoch * 1.2)
    retrain_epochs = max(retrain_epochs, 25)
    retrain_epochs = min(retrain_epochs, 80)

    print(f"\n[Phase 2] Retraining on combined train+val for {retrain_epochs} epochs...")

    model_final = config["class"](**config["kwargs"])

    model_final = retrain_on_train_val(
        model_final,
        train_loader,
        val_loader,
        epochs=retrain_epochs,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        name=f"{model_name.lower()}_final",
        device=DEVICE,
    )

    print(f"\n[Phase 3] Evaluating on test set...")

    test_loss, pred_scaled, true_scaled = evaluate(
        model_final, test_loader, torch.nn.MSELoss(), DEVICE
    )

    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true = target_scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

    all_test_predictions[model_name] = pred

    mse, rmse, mae, mape, r2 = regression_metrics(true, pred)

    print(f"\n{'='*70}")
    print(f"{model_name} RESULTS:")
    print(f"{'='*70}")
    print(f"  RMSE:  ${rmse:>12,.2f}")
    print(f"  MAE:   ${mae:>12,.2f}")
    print(f"  MAPE:  {mape:>11.2f}%")
    print(f"  R²:    {r2:>12.4f}")
    print(f"  Best Epoch (Phase 1): {best_epoch}")
    print(f"  Retrain Epochs: {retrain_epochs}")
    print(f"{'='*70}")

    all_results.append({
        "MODEL": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "best_epoch": best_epoch,
        "retrain_epochs": retrain_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "predictions": (true, pred),
    })


print("\n" + "="*70)
print("=== ENSEMBLE PREDICTIONS ===")
print("="*70)

ensemble_pred = np.mean([all_test_predictions[m] for m in all_test_predictions], axis=0)
mse_ens, rmse_ens, mae_ens, mape_ens, r2_ens = regression_metrics(true, ensemble_pred)

print(f"Ensemble (Simple Average):")
print(f"  RMSE:  ${rmse_ens:>12,.2f}")
print(f"  MAE:   ${mae_ens:>12,.2f}")
print(f"  MAPE:  {mape_ens:>11.2f}%")
print(f"  R²:    {r2_ens:>12.4f}")

all_results.append({
    "MODEL": "Ensemble",
    "RMSE": rmse_ens,
    "MAE": mae_ens,
    "MAPE": mape_ens,
    "R2": r2_ens,
    "best_epoch": 0,
    "retrain_epochs": 0,
    "train_losses": [],
    "val_losses": [],
    "predictions": (true, ensemble_pred),
})


print("\n" + "="*70)
print("=== FINAL COMPARISON ===")
print("="*70)
print(f"{'Model':<15} {'RMSE ($)':<15} {'MAE ($)':<15} {'MAPE (%)':<12} {'R²':<10}")
print("-" * 70)

for res in all_results:
    print(f"{res['MODEL']:<15} {res['RMSE']:>13,.2f}  "
          f"{res['MAE']:>13,.2f}  {res['MAPE']:>10.2f}  {res['R2']:>8.4f}")

print("="*70)

results_df = pd.DataFrame([{
    "MODEL": r["MODEL"],
    "RMSE": r["RMSE"],
    "MAE": r["MAE"],
    "MAPE": r["MAPE"],
    "R2": r["R2"],
    "Best_Epoch": r["best_epoch"],
    "Retrain_Epochs": r["retrain_epochs"]
} for r in all_results])

best_idx = results_df['RMSE'].idxmin()
best_model = results_df.loc[best_idx, 'MODEL']
best_rmse = results_df.loc[best_idx, 'RMSE']
best_r2 = results_df.loc[best_idx, 'R2']

print(f"\nBest Model: {best_model}")
print(f"   RMSE: ${best_rmse:,.2f}")
print(f"   R²:   {best_r2:.4f}")
print("="*70)

results_df.to_csv('final_results.csv', index=False)
print(f"\nResults saved: final_results.csv")

best_result = all_results[best_idx]
pred_df = pd.DataFrame({
    'True_Price': best_result['predictions'][0],
    'Predicted_Price': best_result['predictions'][1],
    'Error': best_result['predictions'][1] - best_result['predictions'][0],
    'Percent_Error': ((best_result['predictions'][1] - best_result['predictions'][0]) /
                      best_result['predictions'][0] * 100)
})
pred_df.to_csv(f'{best_model.lower()}_predictions.csv', index=False)
print(f"{best_model} predictions saved")

print("\n" + "="*70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*70)

if best_rmse < 10000:
    print("\n✓ Excellent performance! RMSE < $10,000")
elif best_rmse < 12000:
    print("\n✓ Good performance! RMSE < $12,000")
else:
    print("\n→ Consider additional hyperparameter tuning")


# ========================
# Create Plots
# ========================

plot_models = ["ResidualMLP", "LSTM", "Transformer"]

# Filter results in the same order
filtered = [r for r in all_results if r["MODEL"] in plot_models]

results_dict = {
    "models": [r["MODEL"] for r in filtered],
    "train_losses": [r["train_losses"] for r in filtered],
    "val_losses": [r["val_losses"] for r in filtered],
    "predictions": [r["predictions"] for r in filtered],
    "results_df": results_df,
}

create_all_plots(results_dict, save_path="plots")