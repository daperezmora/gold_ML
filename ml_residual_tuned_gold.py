# ml_residual_tuned_gold.py
# -*- coding: utf-8 -*-
"""
Baseline GOLD + Residual ML (RF vs Ridge) con validación temporal y figuras doradas.
Entradas:
  - LCS_SIMAT_MarJul2025.csv  (salida de fusion_ml.py)
Salidas en compare_out_gold_tuned/:
  - O3_Calibrado_BaselineResidual_Tuned.csv
  - O3_metrics_summary_tuned.txt
  - figs/*.png  (scatter y timeseries, dorado, fondo transparente)

Requisitos:
  pip install pandas numpy scikit-learn matplotlib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ==================== CONFIG ====================
TARGET = "O3"  # 'O3' o 'CO'
DATA_PATH = Path("LCS_SIMAT_MarJul2025.csv")
OUT_DIR = Path("compare_out_gold_tuned"); OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figs"; FIG_DIR.mkdir(exist_ok=True)

# GOLD theme (fondo transparente, todo dorado)
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

def timeseries_gold(ts, ref, base, ml, ylabel, out_name, title):
    plt.figure(figsize=(10, 5.5), dpi=150)
    plt.grid(True, linestyle="--", linewidth=0.8)
    plt.plot(ts, ref,  color=TXT,       linewidth=2.0, label="Reference")
    plt.plot(ts, base, color=GOLD_DIM,  linewidth=2.0, label="Baseline")
    plt.plot(ts, ml,   color=GOLD,      linewidth=2.2, label="Baseline + Residual ML (Best)")
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel(ylabel)
    legend_gold(loc="best")
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name); plt.close()

# Fórmula GOLD (O3) – compensación exacta
def gold_compensate_vector(o3_raw, temp, rh):
    return (o3_raw / 1.451) + (0.1034*(temp**2) - 2.225*temp + 0.01223*(rh**2) - 1.984*rh + 79.0)

# ==================== CARGA ====================
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])

if TARGET.upper() == "O3":
    y_col   = "O3_ref"                         # ppb
    feats   = ["O3_lcs_ppb", "Temp_lcs", "RH_lcs"]
    y_label = "O₃ (ppb)"
    use_gold = True
elif TARGET.upper() == "CO":
    y_col   = "CO_ref"                         # ppm
    feats   = ["CO_lcs_ppm", "Temp_lcs", "RH_lcs"]
    y_label = "CO (ppm)"
    use_gold = False  # sin fórmula GOLD específica para CO
else:
    raise ValueError("TARGET debe ser 'O3' o 'CO'.")

# ==================== FEATURES (rolling 3h) ====================
df_fe = df.copy()
for c in feats:
    df_fe[f"{c}_rm3h"] = df_fe[c].rolling(window=3, min_periods=1).mean()

# Set con referencia para train/test
dfm = df_fe.dropna(subset=[y_col]).copy()
dfm["month"] = dfm["Timestamp"].dt.month
train_mask = dfm["month"].isin([3,4,5,6])    # mar–jun
test_mask  = dfm["month"].isin([7])          # jul

# Campos base + rolling
feats_ext = feats + [f"{feats[0]}_rm3h", "Temp_lcs_rm3h", "RH_lcs_rm3h"] if TARGET=="O3" \
            else feats + [f"{feats[0]}_rm3h", "Temp_lcs_rm3h", "RH_lcs_rm3h"]

X_train_all = dfm.loc[train_mask, feats_ext]
y_train     = dfm.loc[train_mask, y_col]
X_test_all  = dfm.loc[test_mask,  feats_ext]
y_test      = dfm.loc[test_mask,  y_col]

# ==================== BASELINE (GOLD para O3, Lineal para CO) ====================
if use_gold:
    # imputación para aplicar GOLD
    imp_gold = SimpleImputer(strategy="median")
    Xg_tr = pd.DataFrame(imp_gold.fit_transform(dfm.loc[train_mask, feats]),
                         columns=feats, index=dfm.loc[train_mask].index)
    Xg_te = pd.DataFrame(imp_gold.transform(dfm.loc[test_mask, feats]),
                         columns=feats, index=dfm.loc[test_mask].index)

    # compensación GOLD
    o3c_tr = gold_compensate_vector(Xg_tr.iloc[:,0], Xg_tr.iloc[:,1], Xg_tr.iloc[:,2])
    o3c_te = gold_compensate_vector(Xg_te.iloc[:,0], Xg_te.iloc[:,1], Xg_te.iloc[:,2])

    # calibración lineal final SOLO con train: y_base = alpha + beta * O3_comp
    lin_cal = LinearRegression()
    lin_cal.fit(o3c_tr.to_numpy().reshape(-1,1), y_train.to_numpy())
    yb_tr = lin_cal.predict(o3c_tr.to_numpy().reshape(-1,1))
    yb_te = lin_cal.predict(o3c_te.to_numpy().reshape(-1,1))
    alpha, beta = float(lin_cal.intercept_), float(lin_cal.coef_[0])
    print("\n[GOLD Baseline] alpha, beta =", alpha, beta)
else:
    # CO: baseline lineal con imputación
    base_lin = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("lin", LinearRegression())])
    base_lin.fit(dfm.loc[train_mask, feats], y_train)
    yb_tr = base_lin.predict(dfm.loc[train_mask, feats])
    yb_te = base_lin.predict(dfm.loc[test_mask,  feats])
    print("\n[Linear Baseline] (intercept, coefs) =",
          base_lin.named_steps["lin"].intercept_, *base_lin.named_steps["lin"].coef_)

# ==================== RESIDUAL LEARNING (RF vs Ridge) ====================
# Residuo = y_ref - y_baseline
res_tr = y_train - yb_tr

# Dos variantes de features para el residuo (elige la mejor por CV temporal):
# v1: sin y_baseline como feature
Xtr_v1 = X_train_all.copy()
Xte_v1 = X_test_all.copy()

# v2: con y_baseline como feature adicional
Xtr_v2 = X_train_all.copy(); Xtr_v2["y_baseline"] = yb_tr
Xte_v2 = X_test_all.copy();  Xte_v2["y_baseline"] = yb_te

candidates = [
    ("RF_v1", Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))])),
    ("RF_v2", Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))])),
    ("Ridge_v1", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler(with_mean=False)),
                           ("rg", Ridge(random_state=42))])),
    ("Ridge_v2", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler(with_mean=False)),
                           ("rg", Ridge(random_state=42))])),
]

param_grids = {
    "RF_v1": {
        "rf__n_estimators": [150, 250],
        "rf__max_depth": [6, 8],
        "rf__min_samples_leaf": [3, 5],
        "rf__max_features": ["sqrt", 0.8],
    },
    "RF_v2": {
        "rf__n_estimators": [150, 250],
        "rf__max_depth": [6, 8],
        "rf__min_samples_leaf": [3, 5],
        "rf__max_features": ["sqrt", 0.8],
    },
    "Ridge_v1": { "rg__alpha": [0.5, 1.0, 2.0, 5.0] },
    "Ridge_v2": { "rg__alpha": [0.5, 1.0, 2.0, 5.0] },
}

# Validación temporal (en TRAIN mar–jun)
tscv = TimeSeriesSplit(n_splits=4)
best_name, best_cv_score, best_est, best_variant = None, np.inf, None, None

for name, pipe in candidates:
    grid = param_grids[name]
    # MAE como métrica primaria (minimizar)
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=-1)
    Xuse = Xtr_v1 if name.endswith("v1") else Xtr_v2
    gs.fit(Xuse, res_tr)
    mae_cv = -gs.best_score_
    print(f"[CV] {name}: best MAE={mae_cv:.3f} with params={gs.best_params_}")
    if mae_cv < best_cv_score:
        best_cv_score = mae_cv
        best_name = name
        best_est = gs.best_estimator_
        best_variant = "v1" if name.endswith("v1") else "v2"

print(f"\n>> Mejor modelo residual (CV temporal): {best_name} | MAE={best_cv_score:.3f}")

# Entrenar el mejor sobre TODO el TRAIN y evaluar en TEST
Xtr_best = Xtr_v1 if best_variant=="v1" else Xtr_v2
Xte_best = Xte_v1 if best_variant=="v1" else Xte_v2

best_est.fit(Xtr_best, res_tr)
res_hat_tr = best_est.predict(Xtr_best)
res_hat_te = best_est.predict(Xte_best)

y_hat_tr = yb_tr + res_hat_tr
y_hat_te = yb_te + res_hat_te

# ==================== MÉTRICAS ====================
m_base = metrics_block("Baseline", y_train, yb_tr, y_test, yb_te)
m_ml   = metrics_block(f"Baseline + Residual ML (Best: {best_name})", y_train, y_hat_tr, y_test, y_hat_te)

# ==================== FIGURAS (GOLD) ====================
scatter_gold(
    y_true=y_test, y_pred=yb_te,
    title=f"{TARGET} — GOLD Baseline + Calibration (Test)" if use_gold else f"{TARGET} — Baseline (Test)",
    xlabel=f"{TARGET} Reference {y_label[y_label.find('('):]}",
    ylabel=f"{TARGET} Baseline {y_label[y_label.find('('):]}",
    out_name=f"{TARGET.lower()}_scatter_gold_baseline_cal_tuned.png"
)

scatter_gold(
    y_true=y_test, y_pred=y_hat_te,
    title=f"{TARGET} — Baseline + Residual ML (Best: {best_name})",
    xlabel=f"{TARGET} Reference {y_label[y_label.find('('):]}",
    ylabel=f"{TARGET} Pred (Baseline+ML) {y_label[y_label.find('('):]}",
    out_name=f"{TARGET.lower()}_scatter_gold_baseline_residual_tuned.png"
)

jul_mask = dfm["month"].eq(7)
timeseries_gold(
    ts=dfm.loc[jul_mask,"Timestamp"],
    ref=dfm.loc[jul_mask, y_col],
    base=yb_te,
    ml=y_hat_te,
    ylabel=y_label,
    out_name=f"{TARGET.lower()}_timeseries_july_gold_tuned.png",
    title="Time Series — July (GOLD Baseline + Residual ML Tuned)"
)

# ==================== EXPORTACIÓN COMPLETA ====================
df_out = df_fe.copy()
# Baseline a toda la serie:
mask_full_base = df_out[feats].notna().any(axis=1)
if use_gold:
    imp_full = SimpleImputer(strategy="median")
    Xfull_imp = pd.DataFrame(imp_full.fit_transform(df_out.loc[mask_full_base, feats]),
                             columns=feats, index=df_out.loc[mask_full_base].index)
    o3c_full = gold_compensate_vector(Xfull_imp.iloc[:,0], Xfull_imp.iloc[:,1], Xfull_imp.iloc[:,2])
    # usar alpha/beta del TRAIN
    df_out.loc[mask_full_base, f"{TARGET}_baseline_cal"] = (alpha + beta * o3c_full.to_numpy())
else:
    base_lin_full = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("lin", LinearRegression())])
    base_lin_full.fit(dfm.loc[train_mask, feats], y_train)
    df_out.loc[mask_full_base, f"{TARGET}_baseline_cal"] = base_lin_full.predict(df_out.loc[mask_full_base, feats])

# Residual ML (mejor) a toda la serie:
# construir las mismas features que en best_variant
for c in feats:
    df_out[f"{c}_rm3h"] = df_out[c].rolling(window=3, min_periods=1).mean()

mask_full_ml = df_out[feats_ext].notna().any(axis=1)
Xfull_best = df_out.loc[mask_full_ml, feats_ext].copy()
if best_variant == "v2":
    Xfull_best["y_baseline"] = df_out.loc[mask_full_ml, f"{TARGET}_baseline_cal"]

df_out.loc[mask_full_ml, f"{TARGET}_ml_residual_cal"] = best_est.predict(Xfull_best)

out_csv = OUT_DIR / f"{TARGET}_Calibrado_BaselineResidual_Tuned.csv"
df_out.to_csv(out_csv, index=False)

# Métricas resumen
summary_txt = OUT_DIR / f"{TARGET}_metrics_summary_tuned.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(f"TARGET = {TARGET}\n")
    f.write(f"Best residual model: {best_name}\n\n")
    f.write("Baseline\n")
    f.write(f"Train: R2={m_base['r2_tr']:.3f} RMSE={m_base['rmse_tr']:.2f} MAE={m_base['mae_tr']:.2f}\n")
    f.write(f"Test : R2={m_base['r2_te']:.3f} RMSE={m_base['rmse_te']:.2f} MAE={m_base['mae_te']:.2f}\n\n")
    f.write(f"Baseline + Residual ML (Best: {best_name})\n")
    f.write(f"Train: R2={m_ml['r2_tr']:.3f} RMSE={m_ml['rmse_tr']:.2f} MAE={m_ml['mae_tr']:.2f}\n")
    f.write(f"Test : R2={m_ml['r2_te']:.3f} RMSE={m_ml['rmse_te']:.2f} MAE={m_ml['mae_te']:.2f}\n")

print(f"\n✅ Exportado CSV: {out_csv.resolve()}")
print(f"🎨 Figuras GOLD en: {FIG_DIR.resolve()}")
print(f"🧾 Métricas: {summary_txt.resolve()}")
