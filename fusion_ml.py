# -*- coding: utf-8 -*-
"""
Fusión Smability + SIMAT y entrenamiento ML (línea base)
Versión: 0.2 (fix pandas resample y sklearn RMSE)

Requisitos:
    pip install pandas numpy scikit-learn joblib matplotlib
Archivos esperados:
    - Smability: Report(22).csv
    - SIMAT combinado: SIMAT_BJU_2025-03_07.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TARGET = "O3"  # 'O3' o 'CO'
SMABILITY_PATH = Path("Report(22).csv")
SIMAT_PATH = Path("SIMAT_BJU_2025-03_07.csv")
OUT_DATA = Path("LCS_SIMAT_MarJul2025.csv")
OUT_MODEL = Path(f"modelo_cal_{TARGET.lower()}.joblib")
FIG_DIR = Path("figs"); FIG_DIR.mkdir(exist_ok=True)

# ----------- UTILIDADES -----------------
def metrics(y_true, y_pred):
    """Devuelve R2, RMSE, MAE (RMSE por raíz de MSE para compatibilidad)."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # sin squared=...
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def assert_nonempty(name, X, y):
    if len(y) == 0 or (hasattr(X, "__len__") and len(X) == 0):
        raise RuntimeError(f"Conjunto {name} vacío. Revisa fechas/columnas. "
                           f"len(X)={len(X) if hasattr(X,'__len__') else 'n/a'}, len(y)={len(y)}")

# ----------- CARGA SMABILITY -----------
lcs = pd.read_csv(SMABILITY_PATH)

# 1) Formato de tiempo y orden
lcs = lcs.rename(columns={"PM10*":"PM10"})
lcs["Timestamp"] = pd.to_datetime(lcs["Timestamp"], format="%d/%m/%y %H:%M", errors="coerce")
lcs = lcs.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])

# 2) Limpieza mínima de T (outliers)
lcs.loc[(lcs["Temperature"] > 40) | (lcs["Temperature"] < -10), "Temperature"] = np.nan
lcs["Temperature"] = lcs["Temperature"].interpolate(method="linear", limit_direction="both")

# 3) Re-muestreo horario (usar '1h' para evitar FutureWarning)
lcs = lcs.set_index("Timestamp").resample("1h").mean().reset_index()

# 4) Unidades
lcs["CO_lcs_ppm"] = (lcs["Carbon_Monoxide"] / 1000.0) if "Carbon_Monoxide" in lcs.columns else np.nan

# Nombres claros
lcs = lcs.rename(columns={
    "Ozone": "O3_lcs_ppb",
    "PM2.5": "PM2_5",
    "Relative_Humidity": "RH_lcs",
    "Temperature": "Temp_lcs"
})

# ----------- CARGA SIMAT ---------------
ref = pd.read_csv(SIMAT_PATH, parse_dates=["Timestamp"])
ref = ref.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])

# ----------- UNIÓN POR TIME ------------
df = pd.merge(lcs, ref, on="Timestamp", how="left")
df.to_csv(OUT_DATA, index=False)

# ----------- SELECCIÓN DE VARIABLES ----
if TARGET.upper() == "O3":
    y_col = "O3_ref"         # ppb
    x_cols = ["O3_lcs_ppb", "Temp_lcs", "RH_lcs"]
elif TARGET.upper() == "CO":
    y_col = "CO_ref"         # ppm
    x_cols = ["CO_lcs_ppm", "Temp_lcs", "RH_lcs"]
else:
    raise ValueError("TARGET debe ser 'O3' o 'CO'.")

# Sanidad: que existan las columnas
missing = [c for c in x_cols + [y_col] if c not in df.columns]
if missing:
    raise RuntimeError(f"Faltan columnas en el combinado: {missing}. "
                       "Revisa nombres y SIMAT_BJU_2025-03_07.csv.")

# Filtrar filas con y_ref disponible
df_model = df.dropna(subset=[y_col]).copy()
df_model["month"] = df_model["Timestamp"].dt.month

# Partición temporal: train=mar-jun, test=jul
train_mask = df_model["month"].isin([3,4,5,6])
test_mask  = df_model["month"].isin([7])

X_train = df_model.loc[train_mask, x_cols]
y_train = df_model.loc[train_mask, y_col]
X_test  = df_model.loc[test_mask, x_cols]
y_test  = df_model.loc[test_mask, y_col]

# Checks útiles
assert_nonempty("train", X_train, y_train)
assert_nonempty("test", X_test, y_test)

# Pipelines
lin = Pipeline([("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression())])

rf  = Pipeline([("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
                ))])

# Entrenar
lin.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluación
for name, mdl in [("Linear", lin), ("RF", rf)]:
    pred_tr = mdl.predict(X_train)
    pred_te = mdl.predict(X_test)
    r2_tr, rmse_tr, mae_tr = metrics(y_train, pred_tr)
    r2_te, rmse_te, mae_te = metrics(y_test, pred_te)
    print(f"\n[{name}]")
    print(f"Train -> R2={r2_tr:.3f} | RMSE={rmse_tr:.2f} | MAE={mae_tr:.2f}")
    print(f" Test  -> R2={r2_te:.3f} | RMSE={rmse_te:.2f} | MAE={mae_te:.2f}")

# Elegir mejor
best = rf

# Gráficos rápidos
y_hat = best.predict(X_test)

plt.figure()
plt.scatter(y_test, y_hat, alpha=0.5)
plt.xlabel(f"{TARGET} SIMAT (ref)")
plt.ylabel(f"{TARGET} predicho ML")
plt.title(f"Dispersión {TARGET} - Prueba (jul)")
lims = [min(y_test.min(), y_hat.min()), max(y_test.max(), y_hat.max())]
plt.plot(lims, lims)
plt.tight_layout()
plt.savefig(FIG_DIR / f"scatter_{TARGET.lower()}.png", dpi=150)

plt.figure()
plt.plot(df_model.loc[test_mask, "Timestamp"], y_test.values, label="Ref", linewidth=1.5)
plt.plot(df_model.loc[test_mask, "Timestamp"], y_hat, label="Pred", linewidth=1.0)
plt.legend()
plt.title(f"Serie de tiempo {TARGET} - Prueba (jul)")
plt.xlabel("Tiempo"); plt.ylabel("ppb" if TARGET.upper()=="O3" else "ppm")
plt.tight_layout()
plt.savefig(FIG_DIR / f"timeseries_{TARGET.lower()}.png", dpi=150)

# Guardar modelo
dump(best, OUT_MODEL)
print(f"\n✅ Modelo guardado en: {OUT_MODEL.resolve()}")
print(f"✅ Dataset combinado en: {OUT_DATA.resolve()}")
