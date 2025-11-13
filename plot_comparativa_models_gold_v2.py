# plot_comparativa_models_gold_v2.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from numpy.polynomial.polynomial import polyfit
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# ---------------- CONFIG ----------------
DATA_PATH = Path("compare_RF_vs_SVR_gold/O3_RF_vs_SVR_Calibrado.csv")
OUT_FIG   = Path("compare_RF_vs_SVR_gold/figs/O3_comparativa_GOLD_RF_SVR_v2.png")

# GOLD theme
GOLD       = "#D6B676"   # texto / ejes
GOLD_DIM   = "#8F7E56"   # GOLD baseline
GOLD_RF    = "#C19B59"   # RF
GOLD_SVR   = "#FFD187"   # SVR
GRID_COL   = "#8F7E56"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": GOLD,
    "axes.labelcolor": GOLD,
    "xtick.color": GOLD,
    "ytick.color": GOLD,
    "text.color": GOLD,
    "grid.color": GRID_COL,
    "grid.alpha": 0.35,
    "axes.titleweight": "bold",
    "font.family": "sans-serif",
    "font.size": 14,
    "savefig.transparent": True,
})

def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

# ---------------- LOAD ----------------
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

# Nos quedamos con julio, como en los otros análisis
df = df.copy()
df["month"] = df["Timestamp"].dt.month
df_jul = df[df["month"] == 7].dropna(subset=["O3_ref"])

y_ref   = df_jul["O3_ref"]
y_gold  = df_jul["O3_baseline_cal"]
res_rf  = df_jul["O3_rf_residual_cal"]
res_svr = df_jul["O3_svr_residual_cal"]

# Predicciones finales de cada modelo
y_rf  = y_gold + res_rf
y_svr = y_gold + res_svr

# ---------------- MÉTRICAS ----------------
def stats(name, yhat):
    r2  = r2_score(y_ref, yhat)
    rm  = rmse(y_ref, yhat)
    mae = mean_absolute_error(y_ref, yhat)
    print(f"{name}: R2={r2:.3f} | RMSE={rm:.2f} | MAE={mae:.2f}")
    return r2, rm, mae

r2_g, _, _  = stats("GOLD", y_gold)
r2_rf, _, _ = stats("RF residual", y_rf)
r2_sv, _, _ = stats("SVR residual", y_svr)   # <- OJO: r2_sv

# ---------------- PLOT ----------------
plt.figure(figsize=(8.5, 5.5), dpi=150)
plt.grid(True, linestyle="--", linewidth=0.8)

# Nubes de puntos
plt.scatter(y_ref, y_gold, s=16, alpha=0.5, color=GOLD_DIM)
plt.scatter(y_ref, y_rf,   s=16, alpha=0.5, color=GOLD_RF)
plt.scatter(y_ref, y_svr,  s=16, alpha=0.5, color=GOLD_SVR)

# Línea 1:1
lo = float(min(y_ref.min(), y_gold.min(), y_rf.min(), y_svr.min()))
hi = float(max(y_ref.max(), y_gold.max(), y_rf.max(), y_svr.max()))
plt.plot([lo, hi], [lo, hi], linestyle="--", color=GOLD, linewidth=2.0)

# --------------------------------------------
# LÍNEAS DE TENDENCIA (con estilos distintos)
# --------------------------------------------
xs = np.linspace(lo, hi, 200)

# GOLD (dash-dot)
b_g, m_g = polyfit(y_ref, y_gold, 1)
ys = m_g*xs + b_g
line_gold, = plt.plot(xs, ys, linestyle="-.", color=GOLD_DIM, linewidth=2.0)

# RF (dotted)
b_rf, m_rf = polyfit(y_ref, y_rf, 1)
ys = m_rf*xs + b_rf
line_rf, = plt.plot(xs, ys, linestyle=":", color=GOLD_RF, linewidth=2.0)

# SVR (solid)
b_svr, m_svr = polyfit(y_ref, y_svr, 1)
ys = m_svr*xs + b_svr
line_svr, = plt.plot(xs, ys, linestyle="-", color=GOLD_SVR, linewidth=2.5)

# --------------------------------------------
# CREAR LEYENDA FINAL COMBINADA
# --------------------------------------------
# Puntos de ejemplo
pt_gold = Line2D([0], [0], marker='o', color='none', markerfacecolor=GOLD_DIM,
                 markersize=7, alpha=0.7)
pt_rf   = Line2D([0], [0], marker='o', color='none', markerfacecolor=GOLD_RF,
                 markersize=7, alpha=0.7)
pt_svr  = Line2D([0], [0], marker='o', color='none', markerfacecolor=GOLD_SVR,
                 markersize=7, alpha=0.7)

# Líneas de ejemplo
ln_gold = Line2D([0], [0], color=GOLD_DIM, linestyle="-.", linewidth=2)
ln_rf   = Line2D([0], [0], color=GOLD_RF, linestyle=":",  linewidth=2)
ln_svr  = Line2D([0], [0], color=GOLD_SVR, linestyle="-", linewidth=2.5)
ln_perf = Line2D([0], [0], color=GOLD, linestyle="--", linewidth=2)

labels = [
    f"GOLD (R²={r2_g:.2f})",
    f"RF (R²={r2_rf:.2f})",
    f"SVR (R²={r2_sv:.2f})",   # <- aquí usamos r2_sv
    "Perfecta correlación"
]

legend_handles = [ln_gold, ln_rf, ln_svr, ln_perf]

leg = plt.legend(handles=legend_handles, labels=labels,
                 frameon=True, fontsize=12, loc="upper left")
leg.get_frame().set_edgecolor(GOLD)
leg.get_frame().set_facecolor("none")

plt.title("Comparación de modelos de calibración (julio 2025)")
plt.xlabel("O₃ referencia SIMAT (ppb)")
plt.ylabel("O₃ modelo (ppb)")

plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG)
plt.close()

print("Figura guardada en:", OUT_FIG)
