# fusion_ml_baseline_residual_gold.py
# -*- coding: utf-8 -*-
"""
Baseline GOLD + Residual ML con tema dorado (transparente)
- Carga LCS+SIMAT (mar–jul 2025)
- Para O3: usa la fórmula GOLD exacta de compensación + calibración lineal final (alpha, beta) aprendida SOLO con TRAIN
- Para CO: baseline lineal con imputación (no hay fórmula GOLD específica)
- Residual ML: Random Forest sobre (y_ref - y_baseline)
- Exporta figuras en tema GOLD y un CSV con las columnas calibradas

Requisitos:
    pip install pandas numpy scikit-learn matplotlib
Entrada:
    - LCS_SIMAT_MarJul2025.csv (generado por fusion_ml.py)
Salidas (en compare_out_gold/):
    - <TARGET>_Calibrado_BaselineResidual.csv
    - <TARGET>_metrics_summary.txt
    - figs/*.png  (fondo transparente, tonos dorados)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==================== CONFIG ====================
TARGET = "O3"  # 'O3' o 'CO'
DATA_PATH = Path("LCS_SIMAT_MarJul2025.csv")
OUT_DIR = Path("compare_out_gold"); OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figs"; FIG_DIR.mkdir(exist_ok=True)

# GOLD theme (idéntico a tu script GOLD, fondo transparente)
GOLD       = "#D6B676"   # principal
GOLD_LIGHT = "#E0C78A"   # puntos
GOLD_DIM   = "#8F7E56"   # grid / línea y=x
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

# ==================== UTILIDADES ====================
def rmse(y, yhat) -> float:
    return float(np.sqrt(mean_squared_error(y, yhat)))

def metrics_block(name, y_tr, yhat_tr, y_te, yhat_te):
    r2_tr, r2_te = r2_score(y_tr, yhat_tr), r2_score(y_te, yhat_te)
    rmse_tr, rmse_te = rmse(y_tr, yhat_tr), rmse(y_te, yhat_te)
    mae_tr, mae_te = mean_absolute_error(y_tr, yhat_tr), mean_absolute_error(y_te, yhat_te)
    print(f"\n[{name}]")
    print(f"Train -> R2={r2_tr:.3f} | RMSE={rmse_tr:.2f} | MAE={mae_tr:.2f}")
    print(f" Test  -> R2={r2_te:.3f} | RMSE={rmse_te:.2f} | MAE={mae_te:.2f}")
    return dict(r2_tr=r2_tr, rmse_tr=rmse_tr, mae_tr=mae_tr, r2_te=r2_te, rmse_te=rmse_te, mae_te=mae_te)

def legend_gold(loc="best"):
    leg = plt.legend(loc=loc, frameon=True)
    leg.get_frame().set_edgecolor(GOLD)
    leg.get_frame().set_facecolor("none")
    for t in leg.get_texts():
        t.set_color(TXT)

def scatter_gold(y_true, y_pred, title, xlabel, ylabel, out_name):
    plt.figure(figsize=(8.5, 5.5), dpi=150)
    plt.grid(True, linestyle="--", linewidth=0.8)
    plt.scatter(y_true, y_pred, s=18, color=GOLD_LIGHT, edgecolor="none", alpha=0.75)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color=GOLD_DIM, linewidth=2.0, label="Perfect Correlation")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    legend_gold(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name); plt.close()

def timeseries_gold(ts, ref, base, ml, ylabel, out_name, title="Time Series — July"):
    plt.figure(figsize=(10, 5.5), dpi=150)
    plt.grid(True, linestyle="--", linewidth=0.8)
    plt.plot(ts, ref,  color=TXT,       linewidth=2.0, label="Reference")
    plt.plot(ts, base, color=GOLD_DIM,  linewidth=2.0, label="Baseline")
    plt.plot(ts, ml,   color=GOLD,      linewidth=2.2, label="Baseline + Residual ML")
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel(ylabel)
    legend_gold(loc="best")
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name); plt.close()

# Versión vectorizada de tu GOLD compensate (solo O3)
def gold_compensate_vector(o3_raw, temp, rh):
    """
    O3_comp = (o3/1.451) + 0.1034*T^2 - 2.225*T + 0.01223*RH^2 - 1.984*RH + 79
    - o3_raw en ppb, temp en °C, rh en %
    """
    return (o3_raw / 1.451) + (0.1034*(temp**2) - 2.225*temp + 0.01223*(rh**2) - 1.984*rh + 79.0)

# ==================== CARGA ====================
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])

if TARGET.upper() == "O3":
    y_col   = "O3_ref"                         # ppb
    feats   = ["O3_lcs_ppb", "Temp_lcs", "RH_lcs"]  # señal cruda + met del LCS
    y_label = "O₃ (ppb)"
    use_gold = True
elif TARGET.upper() == "CO":
    y_col   = "CO_ref"                         # ppm (SIMAT)
    feats   = ["CO_lcs_ppm", "Temp_lcs", "RH_lcs"]
    y_label = "CO (ppm)"
    use_gold = False  # no hay fórmula GOLD específica proveída para CO
else:
    raise ValueError("TARGET debe ser 'O3' o 'CO'.")

# Filtrar filas con referencia disponible
dfm = df.dropna(subset=[y_col]).copy()
dfm["month"] = dfm["Timestamp"].dt.month
train_mask = dfm["month"].isin([3,4,5,6])    # mar–jun
test_mask  = dfm["month"].isin([7])          # jul

X_train, y_train = dfm.loc[train_mask, feats], dfm.loc[train_mask, y_col]
X_test,  y_test  = dfm.loc[test_mask,  feats], dfm.loc[test_mask,  y_col]

if len(X_train)==0 or len(X_test)==0:
    raise RuntimeError("Conjunto train/test vacío. Revisa columnas y meses.")

# ==================== BASELINE ====================
# Para O3: GOLD compensate + calibración lineal (alpha, beta) aprendida con TRAIN
# Para CO: baseline lineal con imputación
if use_gold:
    # 1) imputación para poder aplicar GOLD (sin NaN en raw/T/RH)
    imp_gold = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(imp_gold.fit_transform(X_train), columns=feats, index=X_train.index)
    Xte_imp = pd.DataFrame(imp_gold.transform(X_test),      columns=feats, index=X_test.index)

    # 2) compensación GOLD exacta
    o3c_tr = gold_compensate_vector(Xtr_imp.iloc[:,0], Xtr_imp.iloc[:,1], Xtr_imp.iloc[:,2])
    o3c_te = gold_compensate_vector(Xte_imp.iloc[:,0], Xte_imp.iloc[:,1], Xte_imp.iloc[:,2])

    # 3) calibración lineal final SOLO con TRAIN: y_base = alpha + beta * O3_comp
    lin_cal = LinearRegression()
    lin_cal.fit(o3c_tr.to_numpy().reshape(-1,1), y_train.to_numpy())
    yb_tr = lin_cal.predict(o3c_tr.to_numpy().reshape(-1,1))
    yb_te = lin_cal.predict(o3c_te.to_numpy().reshape(-1,1))
    alpha, beta = float(lin_cal.intercept_), float(lin_cal.coef_[0])
    print("\n[GOLD Baseline] alpha, beta =", alpha, beta)

    # guardamos objetos para el pase completo
    baseline_is_pipeline = False
else:
    # CO u otro target: baseline lineal con imputación
    baseline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("lin", LinearRegression())
    ])
    baseline.fit(X_train, y_train)
    yb_tr = baseline.predict(X_train)
    yb_te = baseline.predict(X_test)
    print("\n[Linear Baseline] (intercept, coefs) =",
          baseline.named_steps["lin"].intercept_, *baseline.named_steps["lin"].coef_)
    baseline_is_pipeline = True

# ==================== RESIDUAL ML ====================
# Residuo = y_ref - y_baseline
res_tr = y_train - yb_tr

rf_residual = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rf",  RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))
])

# Features para residuo: (raw, T, RH) + y_baseline
X_train_res = X_train.copy(); X_train_res["y_baseline"] = yb_tr
X_test_res  = X_test.copy();  X_test_res["y_baseline"]  = yb_te

rf_residual.fit(X_train_res, res_tr)

# Predicción final en TEST
res_hat_te = rf_residual.predict(X_test_res)
y_hat_te   = yb_te + res_hat_te

# ==================== MÉTRICAS ====================
m_base = metrics_block("Baseline", y_train, yb_tr, y_test, yb_te)
m_ml   = metrics_block("Baseline + Residual ML", y_train, yb_tr + rf_residual.predict(X_train_res), y_test, y_hat_te)

# ==================== FIGURAS (GOLD, transparent) ====================
# 1) "Correlation After GOLD Baseline + Calibration" (para O3 coincide con baseline)
scatter_gold(
    y_true=y_test,
    y_pred=yb_te,
    title=f"{TARGET} — GOLD Baseline + Calibration (Test)" if use_gold else f"{TARGET} — Baseline (Test)",
    xlabel=f"{TARGET} Reference {y_label[y_label.find('('):]}",
    ylabel=f"{TARGET} Baseline {y_label[y_label.find('('):]}",
    out_name=f"{TARGET.lower()}_scatter_gold_baseline_cal.png"
)

# 2) Baseline + Residual ML
scatter_gold(
    y_true=y_test,
    y_pred=y_hat_te,
    title=f"{TARGET} — Baseline + Residual ML (Test)",
    xlabel=f"{TARGET} Reference {y_label[y_label.find('('):]}",
    ylabel=f"{TARGET} Pred (Baseline+ML) {y_label[y_label.find('('):]}",
    out_name=f"{TARGET.lower()}_scatter_gold_baseline_residual.png"
)

# 3) Serie de tiempo julio (1:1 en timestamps con referencia)
jul_mask = dfm["month"].eq(7)
ts_jul   = dfm.loc[jul_mask, "Timestamp"]
ref_jul  = dfm.loc[jul_mask, y_col]
base_jul = yb_te
ml_jul   = y_hat_te
timeseries_gold(
    ts=ts_jul, ref=ref_jul, base=base_jul, ml=ml_jul,
    ylabel=y_label,
    out_name=f"{TARGET.lower()}_timeseries_july_gold.png",
    title="Time Series — July (After GOLD Baseline + Residual ML)" if use_gold else "Time Series — July"
)

# ==================== EXPORTACIÓN COMPLETA ====================
df_out = df.copy()
mask_full = df_out[feats].notna().any(axis=1)  # predecimos donde exista algo; los pipelines imputan

if use_gold:
    # aplicar GOLD + calibración (alpha, beta) a TODA la serie
    Xfull_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df_out.loc[mask_full, feats]),
        columns=feats, index=df_out.loc[mask_full].index
    )
    o3c_full = gold_compensate_vector(Xfull_imp.iloc[:,0], Xfull_imp.iloc[:,1], Xfull_imp.iloc[:,2])
    df_out.loc[mask_full, f"{TARGET}_baseline_cal"] = alpha + beta * o3c_full.to_numpy()
else:
    df_out.loc[mask_full, f"{TARGET}_baseline_cal"] = baseline.predict(df_out.loc[mask_full, feats])

# Residual ML en TODO el periodo (donde haya features)
Xfull_res = df_out.loc[mask_full, feats].copy()
Xfull_res["y_baseline"] = df_out.loc[mask_full, f"{TARGET}_baseline_cal"]
df_out.loc[mask_full, f"{TARGET}_ml_residual_cal"] = rf_residual.predict(Xfull_res)

out_csv = OUT_DIR / f"{TARGET}_Calibrado_BaselineResidual.csv"
df_out.to_csv(out_csv, index=False)

# Guardar métricas
summary_txt = OUT_DIR / f"{TARGET}_metrics_summary.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(f"TARGET = {TARGET}\n\n")
    f.write("Baseline\n")
    f.write(f"Train: R2={m_base['r2_tr']:.3f} RMSE={m_base['rmse_tr']:.2f} MAE={m_base['mae_tr']:.2f}\n")
    f.write(f"Test : R2={m_base['r2_te']:.3f} RMSE={m_base['rmse_te']:.2f} MAE={m_base['mae_te']:.2f}\n\n")
    f.write("Baseline + Residual ML\n")
    f.write(f"Train: R2={m_ml['r2_tr']:.3f} RMSE={m_ml['rmse_tr']:.2f} MAE={m_ml['mae_tr']:.2f}\n")
    f.write(f"Test : R2={m_ml['r2_te']:.3f} RMSE={m_ml['rmse_te']:.2f} MAE={m_ml['mae_te']:.2f}\n")

print(f"\n✅ Exportado CSV: {out_csv.resolve()}")
print(f"🎨 Figuras GOLD en: {FIG_DIR.resolve()}")
print(f"🧾 Métricas: {summary_txt.resolve()}")
