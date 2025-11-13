# ml_residual_RF_vs_SVR_gold.py
# -*- coding: utf-8 -*-
"""
Comparación Residual ML:
  - Random Forest (RF)
  - Support Vector Regressor (SVR, kernel RBF)
Sobre Baseline GOLD (para O3)

Incluye:
  - Validación temporal
  - Rolling mean 3h
  - Figuras GOLD (fondo transparente)
  - CSV con outputs de ambos métodos
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==================== CONFIG ====================
TARGET = "O3"   # solo O3 tiene fórmula GOLD
DATA_PATH = Path("LCS_SIMAT_MarJul2025.csv")

OUT_DIR = Path("compare_RF_vs_SVR_gold")
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)

# GOLD theme (transparente)
GOLD = "#D6B676"
GOLD_LIGHT = "#E0C78A"
GOLD_DIM = "#8F7E56"
TXT = GOLD
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
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def metrics_block(name, y_tr, yhat_tr, y_te, yhat_te):
    r2_tr, r2_te = r2_score(y_tr, yhat_tr), r2_score(y_te, yhat_te)
    rmse_tr, rmse_te = rmse(y_tr, yhat_tr), rmse(y_te, yhat_te)
    mae_tr, mae_te = mean_absolute_error(y_tr, yhat_tr), mean_absolute_error(y_te, yhat_te)
    print("\n[" + name + "]")
    print(f"Train -> R2={r2_tr:.3f} | RMSE={rmse_tr:.2f} | MAE={mae_tr:.2f}")
    print(f" Test -> R2={r2_te:.3f} | RMSE={rmse_te:.2f} | MAE={mae_te:.2f}")
    return (r2_te, rmse_te, mae_te)

def scatter_gold(y_true, y_pred, title, out_name):
    plt.figure(figsize=(8.5,5.5), dpi=150)
    plt.grid(True, linestyle="--", linewidth=0.8)
    plt.scatter(y_true, y_pred, s=18, color=GOLD_LIGHT, edgecolor="none")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color=GOLD_DIM, linewidth=2.0, label="Perfect Line")
    plt.legend()
    plt.title(title)
    plt.xlabel("Reference (ppb)"); plt.ylabel("Prediction (ppb)")
    plt.tight_layout()
    plt.savefig(FIG_DIR/out_name); plt.close()

def timeseries_gold(ts, ref, base, rf, svr, out_name):
    plt.figure(figsize=(10,5.5), dpi=150)
    plt.grid(True, linestyle="--", linewidth=0.8)
    plt.plot(ts, ref, color=TXT, linewidth=2, label="Reference")
    plt.plot(ts, base, color=GOLD_DIM, linewidth=2, label="Baseline GOLD")
    plt.plot(ts, rf, color="#C19B59", linewidth=2, label="RF Residual")
    plt.plot(ts, svr, color="#FFD187", linewidth=2, label="SVR Residual")
    plt.legend()
    plt.ylabel("O₃ (ppb)")
    plt.title("Time Series — July (RF vs SVR)")
    plt.tight_layout()
    plt.savefig(FIG_DIR/out_name); plt.close()

# ==================== CARGA ====================
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").drop_duplicates("Timestamp")

# columnas
feats = ["O3_lcs_ppb", "Temp_lcs", "RH_lcs"]
y_col = "O3_ref"

# rolling 3h
df["O3_lcs_ppb_rm3h"] = df["O3_lcs_ppb"].rolling(3, min_periods=1).mean()
df["Temp_lcs_rm3h"]   = df["Temp_lcs"].rolling(3, min_periods=1).mean()
df["RH_lcs_rm3h"]     = df["RH_lcs"].rolling(3, min_periods=1).mean()

feats_ext = feats + ["O3_lcs_ppb_rm3h", "Temp_lcs_rm3h", "RH_lcs_rm3h"]

# ==================== TRAIN/TEST ====================
dfm = df.dropna(subset=[y_col]).copy()
dfm["month"] = dfm["Timestamp"].dt.month
train_mask = dfm["month"].isin([3,4,5,6])
test_mask  = dfm["month"].isin([7])

X_train_all = dfm.loc[train_mask, feats_ext]
y_train = dfm.loc[train_mask, y_col]

X_test_all = dfm.loc[test_mask, feats_ext]
y_test = dfm.loc[test_mask, y_col]

# ==================== BASELINE GOLD ====================
def gold_compensate(o3, t, h):
    return (o3/1.451) + (0.1034*t*t - 2.225*t + 0.01223*h*h - 1.984*h + 79)

imp = SimpleImputer(strategy="median")
X_tr_gold = pd.DataFrame(imp.fit_transform(dfm.loc[train_mask, feats]),
                         columns=feats, index=dfm.loc[train_mask].index)
X_te_gold = pd.DataFrame(imp.transform(dfm.loc[test_mask, feats]),
                         columns=feats, index=dfm.loc[test_mask].index)

o3c_tr = gold_compensate(X_tr_gold.iloc[:,0], X_tr_gold.iloc[:,1], X_tr_gold.iloc[:,2])
o3c_te = gold_compensate(X_te_gold.iloc[:,0], X_te_gold.iloc[:,1], X_te_gold.iloc[:,2])

# calibración lineal final
lin_cal = LinearRegression()
lin_cal.fit(o3c_tr.to_numpy().reshape(-1,1), y_train.to_numpy())

alpha, beta = lin_cal.intercept_, lin_cal.coef_[0]
print("\n[GOLD baseline] alpha, beta =", alpha, beta)

yb_tr = lin_cal.predict(o3c_tr.to_numpy().reshape(-1,1))
yb_te = lin_cal.predict(o3c_te.to_numpy().reshape(-1,1))

# RESIDUO
res_tr = y_train - yb_tr

# ==================== MODELOS RESIDUALES ====================
# RF y SVR
models = {
    "RF": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            random_state=42, max_depth=6, min_samples_leaf=5, n_estimators=250, n_jobs=-1))
    ]),
    "SVR": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])
}

# hiperparámetros SVR
svr_params = {
    "svr__C": [1, 5, 10, 50],
    "svr__gamma": ["scale"]
}

# X para residuo = features + y_baseline
X_train_res = X_train_all.copy()
X_train_res["y_baseline"] = yb_tr

X_test_res = X_test_all.copy()
X_test_res["y_baseline"] = yb_te

# Validación temporal
tscv = TimeSeriesSplit(n_splits=4)

# ---- RF (sin grid para rapidez) ----
models["RF"].fit(X_train_res, res_tr)
res_rf_tr = models["RF"].predict(X_train_res)
res_rf_te = models["RF"].predict(X_test_res)

# ---- SVR (con grid search) ----
gs = GridSearchCV(models["SVR"], svr_params, scoring="neg_mean_absolute_error",
                  cv=tscv, n_jobs=-1)
gs.fit(X_train_res, res_tr)
svr_best = gs.best_estimator_
print("\n[SVR] best params:", gs.best_params_)

res_svr_tr = svr_best.predict(X_train_res)
res_svr_te = svr_best.predict(X_test_res)

# ==================== PREDICCIONES FINALES ====================
y_rf_te  = yb_te + res_rf_te
y_svr_te = yb_te + res_svr_te

# ==================== MÉTRICAS ====================
print("\n=== MÉTRICAS ===")
print("\nBaseline GOLD:")
metrics_block("Baseline GOLD", y_train, yb_tr, y_test, yb_te)

print("\nRF residual:")
rf_metrics = metrics_block("RF Residual", y_train, yb_tr + res_rf_tr, y_test, y_rf_te)

print("\nSVR residual:")
svr_metrics = metrics_block("SVR Residual", y_train, yb_tr + res_svr_tr, y_test, y_svr_te)

# ==================== FIGURAS GOLD ====================
scatter_gold(y_test, y_rf_te,  "RF Residual (Test)",  "rf_scatter.png")
scatter_gold(y_test, y_svr_te, "SVR Residual (Test)", "svr_scatter.png")

# serie temporal julio
ts_jul = dfm.loc[test_mask, "Timestamp"]
timeseries_gold(ts_jul, y_test, yb_te, y_rf_te, y_svr_te, "rf_vs_svr_timeseries.png")

# ==================== EXPORTACIÓN ====================
df_out = df.copy()

# aplicar baseline a toda la serie
mask_full = df_out[feats].notna().any(axis=1)
Xfull_imp = pd.DataFrame(imp.transform(df_out.loc[mask_full, feats]),
                         columns=feats, index=df_out.loc[mask_full].index)

o3c_full = gold_compensate(Xfull_imp.iloc[:,0], Xfull_imp.iloc[:,1], Xfull_imp.iloc[:,2])
df_out.loc[mask_full, "O3_baseline_cal"] = alpha + beta * o3c_full.to_numpy()

# rolling 3h también para predicciones completas
for c in feats:
    df_out[c+"_rm3h"] = df_out[c].rolling(3, min_periods=1).mean()

mask_ml = df_out[feats_ext].notna().any(axis=1)
Xfull = df_out.loc[mask_ml, feats_ext].copy()
Xfull["y_baseline"] = df_out.loc[mask_ml, "O3_baseline_cal"]

df_out.loc[mask_ml, "O3_rf_residual_cal"]  = models["RF"].predict(Xfull)
df_out.loc[mask_ml, "O3_svr_residual_cal"] = svr_best.predict(Xfull)

out_csv = OUT_DIR / "O3_RF_vs_SVR_Calibrado.csv"
df_out.to_csv(out_csv, index=False)

# ==================== EXPORTAR RESUMEN TXT ====================
txt_path = OUT_DIR / "O3_RF_vs_SVR_metrics_summary.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=== SUMMARY: Baseline GOLD vs RF vs SVR ===\n\n")

    f.write("Baseline GOLD\n")
    f.write(f"R2_test   = {metrics_block('Baseline GOLD', y_train, yb_tr, y_test, yb_te)[0]:.3f}\n")
    f.write(f"RMSE_test = {metrics_block('Baseline GOLD', y_train, yb_tr, y_test, yb_te)[1]:.2f}\n")
    f.write(f"MAE_test  = {metrics_block('Baseline GOLD', y_train, yb_tr, y_test, yb_te)[2]:.2f}\n\n")

    f.write("Random Forest Residual\n")
    f.write(f"R2_test   = {metrics_block('RF Residual', y_train, yb_tr + res_rf_tr, y_test, y_rf_te)[0]:.3f}\n")
    f.write(f"RMSE_test = {metrics_block('RF Residual', y_train, yb_tr + res_rf_tr, y_test, y_rf_te)[1]:.2f}\n")
    f.write(f"MAE_test  = {metrics_block('RF Residual', y_train, yb_tr + res_rf_tr, y_test, y_rf_te)[2]:.2f}\n\n")

    f.write("SVR Residual (Best Model)\n")
    f.write(f"R2_test   = {metrics_block('SVR Residual', y_train, yb_tr + res_svr_tr, y_test, y_svr_te)[0]:.3f}\n")
    f.write(f"RMSE_test = {metrics_block('SVR Residual', y_train, yb_tr + res_svr_tr, y_test, y_svr_te)[1]:.2f}\n")
    f.write(f"MAE_test  = {metrics_block('SVR Residual', y_train, yb_tr + res_svr_tr, y_test, y_svr_te)[2]:.2f}\n")

print("\nResumen TXT generado:", txt_path)

print("\nCSV generado:", out_csv)
print("Figuras en:", FIG_DIR)
