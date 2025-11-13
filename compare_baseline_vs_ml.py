# -*- coding: utf-8 -*-
"""
Comparación Baseline (tipo claudeSmaGOLD) vs Modelo ML en paleta dorada
Versión: 1.0

Qué hace:
- Carga LCS + SIMAT (mar–jul 2025).
- Baseline: Regresión Lineal con [señal cruda LCS, Temp, RH].
- ML: carga un modelo .joblib (p.ej. RandomForest o Lineal) entrenado antes.
- Evalúa en train (mar–jun) y test (jul).
- Exporta predicciones completas y figuras con estética dorada para póster.

Requisitos:
    pip install pandas numpy scikit-learn joblib matplotlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============ CONFIGURACIÓN ============
TARGET = "O3"  # 'O3' o 'CO'
DATA_PATH = Path("LCS_SIMAT_MarJul2025.csv")
ML_MODEL_PATH = Path("modelo_cal_o3.joblib")  # cambia si usas otro modelo o CO
OUT_DIR = Path("compare_out"); OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figs"; FIG_DIR.mkdir(exist_ok=True)

# Paleta dorada (inspirada en claudeSmaGOLD)
COL_BG   = "#FFFFFF"   # fondo blanco para póster
COL_GOLD = "#D4AF37"   # gold metálico
COL_GOLD2= "#C9A227"   # gold oscuro
COL_AMBR = "#B8860B"   # goldenrod oscuro
COL_GRAY = "#555555"
COL_REF  = "#222222"   # referencia (negro suave)

# ============ UTILIDADES ============
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_block(name, y_tr, yhat_tr, y_te, yhat_te):
    r2_tr, rmse_tr, mae_tr = r2_score(y_tr, yhat_tr), rmse(y_tr, yhat_tr), mean_absolute_error(y_tr, yhat_tr)
    r2_te, rmse_te, mae_te = r2_score(y_te, yhat_te), rmse(y_te, yhat_te), mean_absolute_error(y_te, yhat_te)
    print(f"\n[{name}]")
    print(f"Train -> R2={r2_tr:.3f} | RMSE={rmse_tr:.2f} | MAE={mae_tr:.2f}")
    print(f" Test  -> R2={r2_te:.3f} | RMSE={rmse_te:.2f} | MAE={mae_te:.2f}")
    return {"r2_tr":r2_tr,"rmse_tr":rmse_tr,"mae_tr":mae_tr,
            "r2_te":r2_te,"rmse_te":rmse_te,"mae_te":mae_te}

def gold_style():
    plt.rcParams.update({
        "figure.figsize": (7.5, 5.0),
        "figure.dpi": 150,
        "axes.facecolor": COL_BG,
        "figure.facecolor": COL_BG,
        "axes.edgecolor": COL_GRAY,
        "axes.labelcolor": COL_GRAY,
        "xtick.color": COL_GRAY,
        "ytick.color": COL_GRAY,
        "grid.color": "#E6E6E6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
    })

def save_scatter(y_true, y_pred, title, xlabel, ylabel, out_name, color_line=COL_GOLD, color_pts=COL_AMBR):
    gold_style()
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="none", s=18, color=color_pts)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], color=color_line, linewidth=2.0, label="1:1")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name)
    plt.close()

def save_timeseries_july(ts, y_ref, y_base, y_ml, ylabel, out_name):
    gold_style()
    plt.figure()
    plt.plot(ts, y_ref,  label="Referencia (SIMAT)", color=COL_REF,  linewidth=2.2)
    plt.plot(ts, y_base, label="Baseline (claude)",  color=COL_GOLD2, linewidth=2.0)
    plt.plot(ts, y_ml,   label="Modelo ML",         color=COL_GOLD,  linewidth=2.0)
    plt.title("Serie de tiempo – Julio (comparación)")
    plt.xlabel("Tiempo"); plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name)
    plt.close()

# ============ CARGA Y PREP ============
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])

# Columnas esperadas del LCS (las generaste en fusion_ml.py)
# O3: O3_lcs_ppb, Temp_lcs, RH_lcs  |  CO: CO_lcs_ppm, Temp_lcs, RH_lcs
if TARGET.upper() == "O3":
    y_col   = "O3_ref"
    x_cols  = ["O3_lcs_ppb", "Temp_lcs", "RH_lcs"]
    y_label = "O₃ (ppb)"
elif TARGET.upper() == "CO":
    y_col   = "CO_ref"      # en ppm (SIMAT)
    x_cols  = ["CO_lcs_ppm", "Temp_lcs", "RH_lcs"]  # Smability ya convertido a ppm
    y_label = "CO (ppm)"
else:
    raise ValueError("TARGET debe ser 'O3' o 'CO'.")

# Filas con referencia disponible
dfm = df.dropna(subset=[y_col]).copy()
dfm["month"] = dfm["Timestamp"].dt.month
train_mask = dfm["month"].isin([3,4,5,6])  # mar–jun
test_mask  = dfm["month"].isin([7])        # jul

X_train, y_train = dfm.loc[train_mask, x_cols], dfm.loc[train_mask, y_col]
X_test,  y_test  = dfm.loc[test_mask,  x_cols], dfm.loc[test_mask,  y_col]

if len(X_train)==0 or len(X_test)==0:
    raise RuntimeError("Conjunto train/test vacío. Revisa columnas y meses.")

# ============ BASELINE (tipo claude) ============
# Lineal con imputación mediana (idéntico espíritu a claudeSmaGOLD)
baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])
baseline.fit(X_train, y_train)

yhat_tr_base = baseline.predict(X_train)
yhat_te_base = baseline.predict(X_test)

# ============ ML CARGADO (.joblib) ============
ml_model = load(ML_MODEL_PATH)
yhat_tr_ml = ml_model.predict(X_train)
yhat_te_ml = ml_model.predict(X_test)

# ============ MÉTRICAS ============
m_base = metrics_block("Baseline (claude-style)", y_train, yhat_tr_base, y_test, yhat_te_base)
m_ml   = metrics_block("Modelo ML (.joblib)",     y_train, yhat_tr_ml,   y_test, yhat_te_ml)

# ============ GRÁFICAS DORADAS ============
# Dispersión en test
save_scatter(
    y_true=y_test, y_pred=yhat_te_base,
    title=f"{TARGET} – Dispersión Test (Baseline)",
    xlabel=f"{TARGET} Ref (SIMAT) {y_label[y_label.find('('):]}",
    ylabel=f"{TARGET} Pred Baseline {y_label[y_label.find('('):]}",
    out_name=f"{TARGET.lower()}_scatter_test_baseline_gold.png",
    color_line=COL_GOLD2, color_pts=COL_AMBR
)
save_scatter(
    y_true=y_test, y_pred=yhat_te_ml,
    title=f"{TARGET} – Dispersión Test (ML)",
    xlabel=f"{TARGET} Ref (SIMAT) {y_label[y_label.find('('):]}",
    ylabel=f"{TARGET} Pred ML {y_label[y_label.find('('):]}",
    out_name=f"{TARGET.lower()}_scatter_test_ml_gold.png",
    color_line=COL_GOLD, color_pts=COL_AMBR
)

# Serie de tiempo (julio) Ref vs Baseline vs ML
jul = dfm["month"].eq(7)
ts_jul    = dfm.loc[jul, "Timestamp"]
ref_jul   = dfm.loc[jul, y_col]
base_jul  = baseline.predict(dfm.loc[jul, x_cols])
ml_jul    = ml_model.predict(dfm.loc[jul, x_cols])
save_timeseries_july(ts_jul, ref_jul, base_jul, ml_jul,
                     ylabel=y_label,
                     out_name=f"{TARGET.lower()}_timeseries_july_gold.png")

# ============ EXPORTAR SERIES COMPLETAS ============
# Predicciones para TODO el periodo disponible (no solo julio)
df_out = df.copy()
# Calcula predicciones donde existan features (aunque falte referencia)
mask_X_full = df_out[x_cols].notna().all(axis=1)
df_out.loc[mask_X_full, f"{TARGET}_baseline_cal"] = baseline.predict(df_out.loc[mask_X_full, x_cols])
df_out.loc[mask_X_full, f"{TARGET}_ml_cal"]       = ml_model.predict(df_out.loc[mask_X_full, x_cols])

out_csv = OUT_DIR / f"{TARGET}_Calibrado_Baseline_vs_ML.csv"
df_out.to_csv(out_csv, index=False)
print(f"\n✅ Exportado: {out_csv.resolve()}")

# Guardar un pequeño resumen de métricas
summary_txt = OUT_DIR / f"{TARGET}_metrics_summary.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(f"TARGET={TARGET}\n\n")
    f.write("Baseline (claude-style)\n")
    f.write(f"Train: R2={m_base['r2_tr']:.3f} RMSE={m_base['rmse_tr']:.2f} MAE={m_base['mae_tr']:.2f}\n")
    f.write(f"Test : R2={m_base['r2_te']:.3f} RMSE={m_base['rmse_te']:.2f} MAE={m_base['mae_te']:.2f}\n\n")
    f.write("Modelo ML (.joblib)\n")
    f.write(f"Train: R2={m_ml['r2_tr']:.3f} RMSE={m_ml['rmse_tr']:.2f} MAE={m_ml['mae_tr']:.2f}\n")
    f.write(f"Test : R2={m_ml['r2_te']:.3f} RMSE={m_ml['rmse_te']:.2f} MAE={m_ml['mae_te']:.2f}\n")
print(f"✅ Resumen métricas: {summary_txt.resolve()}")

print("\n🎨 Figuras (paleta dorada) en:", FIG_DIR.resolve())
