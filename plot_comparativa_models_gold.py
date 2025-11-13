# plot_comparativa_models_gold.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------- CONFIG ----------------
DATA_PATH = Path("compare_RF_vs_SVR_gold/O3_RF_vs_SVR_Calibrado.csv")
OUT_FIG   = Path("compare_RF_vs_SVR_gold/figs/O3_comparativa_GOLD_RF_SVR.png")

# GOLD theme (como tus otras figuras)
GOLD       = "#D6B676"
GOLD_LIGHT = "#E0C78A"
GOLD_DIM   = "#8F7E56"
TXT        = GOLD

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": GOLD,
    "axes.labelcolor": TXT,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "text.color": TXT,
    "grid.color": GOLD_DIM,
    "grid.alpha": 0.35,
    "axes.titleweight": "bold",
    "font.family": "sans-serif",
    "font.size": 14,
    "savefig.transparent": True,
})

def rmse(y, yhat):
    return float(np.sqrt(((y - yhat) ** 2).mean()))

# ---------------- LOAD ----------------
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

# Nos quedamos con julio (conjunto de prueba)
df = df.copy()
df["month"] = df["Timestamp"].dt.month
df_jul = df[df["month"] == 7].dropna(subset=["O3_ref"])

y_ref = df_jul["O3_ref"]
y_base = df_jul["O3_baseline_cal"]
y_rf   = df_jul["O3_rf_residual_cal"]
y_svr  = df_jul["O3_svr_residual_cal"]

# Métricas (por si quieres ponerlas en el póster)
def stats(name, yhat):
    r2  = r2_score(y_ref, yhat)
    rm  = rmse(y_ref, yhat)
    mae = np.abs(y_ref - yhat).mean()
    print(f"{name}: R2={r2:.3f} | RMSE={rm:.2f} | MAE={mae:.2f}")
    return r2, rm, mae

r2_b, _, _ = stats("Baseline GOLD", y_base)
r2_rf, _, _ = stats("RF residual",   y_rf)
r2_sv, _, _ = stats("SVR residual",  y_svr)

# ---------------- PLOT ----------------
plt.figure(figsize=(8.5, 5.5), dpi=150)
plt.grid(True, linestyle="--", linewidth=0.8)

# Nubes de puntos
plt.scatter(y_ref, y_base, s=16, alpha=0.5, color=GOLD_DIM,  label=f"GOLD (R²={r2_b:.2f})")
plt.scatter(y_ref, y_rf,   s=16, alpha=0.5, color="#C19B59", label=f"RF (R²={r2_rf:.2f})")
plt.scatter(y_ref, y_svr,  s=16, alpha=0.5, color="#FFD187", label=f"SVR (R²={r2_sv:.2f})")

# Línea 1:1
lo = float(min(y_ref.min(), y_base.min(), y_rf.min(), y_svr.min()))
hi = float(max(y_ref.max(), y_base.max(), y_rf.max(), y_svr.max()))
plt.plot([lo, hi], [lo, hi], linestyle="--", color=GOLD, linewidth=2.0, label="Perfecta correlación")

plt.title("Comparación de modelos de calibración (julio 2025)")
plt.xlabel("O₃ referencia SIMAT (ppb)")
plt.ylabel("O₃ modelo (ppb)")
leg = plt.legend(frameon=True)
leg.get_frame().set_edgecolor(GOLD)
leg.get_frame().set_facecolor("none")

plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG)
plt.close()

print("Figura guardada en:", OUT_FIG)
