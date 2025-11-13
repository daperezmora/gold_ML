# -*- coding: utf-8 -*-
"""
Normalizador SIMAT (Benito Juárez) -> Tablas maestras por hora
Versión: 0.4 (fix combinación de meses con concat)

Qué hace:
- Lee archivos SIMAT con formato "Fecha | Hora | BJU" para CO (ppm), O3 (ppb), RH (%), TMP (°C).
- Corrige encabezados (primera fila es título), convierte "nr" a NaN y a float.
- Construye Timestamp horario (Hora=24 -> 00:00 día siguiente).
- Exporta CSV mensual y un CSV combinado mar–jul (o el rango que tengas).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# =========================
# =======  CONFIG   =======
# =========================
BASE_DIR = Path(".")     # carpeta con tus CSV
YEAR = 2025
MONTHS = ["marzo", "abril", "mayo", "junio", "julio"]

MONTH_TO_MM = {
    "enero": "01","febrero":"02","marzo":"03","abril":"04",
    "mayo":"05","junio":"06","julio":"07","agosto":"08",
    "septiembre":"09","octubre":"10","noviembre":"11","diciembre":"12"
}

VAR_PATTERNS = {
    "CO":  "Promedio horarios de co-{mes}.csv",
    "O3":  "Promedio horarios de o3-{mes}.csv",
    "RH":  "Promedio horarios de rh-{mes}.csv",
    "TMP": "Promedio horarios de tmp-{mes}.csv",
}

OUT_COLS = {"CO":"CO_ref","O3":"O3_ref","RH":"RH_ref","TMP":"Temp_ref"}

# =========================
# =====  UTILIDADES   =====
# =========================
def _read_simat_threecol_robust(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin1", header=None, sep=",")
    mask_fecha = raw.iloc[:,0].astype(str).str.strip().str.lower().eq("fecha")
    header_rows = raw.index[mask_fecha].tolist()
    if not header_rows:
        raise ValueError(f"No encontré encabezados 'Fecha,Hora,BJU' en: {path.name}")
    header_row = header_rows[0]
    headers = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row+1:].copy()
    df.columns = [str(c).strip() for c in headers]
    expected = ["Fecha","Hora","BJU"]
    if not all(col in df.columns for col in expected):
        raise ValueError(f"Encabezados inesperados en {path.name}. Encontré: {df.columns.tolist()}")
    return df[expected]

def _compose_timestamp(fecha_series: pd.Series, hora_series: pd.Series, year: int) -> pd.Series:
    fechas = pd.to_datetime(fecha_series, format="%d/%m/%y", errors="coerce")
    fechas = fechas.map(lambda d: pd.Timestamp(year, d.month, d.day) if pd.notna(d) else pd.NaT)
    horas = pd.to_numeric(hora_series, errors="coerce").astype("Int64")
    add_day = horas.eq(24)
    horas_fix = horas.where(~add_day, 0)
    ts = fechas + pd.to_timedelta(horas_fix.fillna(0), unit="h")
    ts = ts.where(~add_day, ts + pd.to_timedelta(1, unit="D"))
    return ts

def _clean_value_series(s: pd.Series) -> pd.Series:
    s = s.replace({"nr": np.nan, "NR": np.nan, "Na": np.nan, "NA": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")

def parse_simat_variable(path: Path, var_name: str, year: int) -> pd.DataFrame:
    df = _read_simat_threecol_robust(path)
    ts = _compose_timestamp(df["Fecha"], df["Hora"], year)
    val = _clean_value_series(df["BJU"])
    out = pd.DataFrame({"Timestamp": ts, OUT_COLS[var_name]: val})
    return out.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

def integrate_month(month_name: str, base_dir: Path, year: int) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for var_name, pattern in VAR_PATTERNS.items():
        file_path = base_dir / pattern.format(mes=month_name)
        if file_path.exists():
            parts.append(parse_simat_variable(file_path, var_name, year))
        else:
            print(f"⚠️ Falta {file_path.name} (se omite {var_name}).")
    if not parts:
        raise FileNotFoundError(f"Sin archivos SIMAT para '{month_name}' en {base_dir.resolve()}.")
    # Merge por columnas distintas (cada parte tiene distinta columna de valor)
    base = parts[0]
    for piece in parts[1:]:
        base = pd.merge(base, piece, on="Timestamp", how="outer")
    return base.sort_values("Timestamp").reset_index(drop=True)

def save_month(df: pd.DataFrame, year: int, month_name: str, out_dir: Path) -> Path:
    mm = MONTH_TO_MM[month_name.lower()]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"SIMAT_BJU_{year}-{mm}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Guardado: {out_path.name}")
    return out_path

# =========================
# ========= MAIN ==========
# =========================
if __name__ == "__main__":
    out_dir = BASE_DIR

    monthly_tables: List[pd.DataFrame] = []
    for m in MONTHS:
        try:
            print(f"\n📦 Integrando mes: {m} ({YEAR}) ...")
            df_m = integrate_month(m, BASE_DIR, YEAR)
            monthly_tables.append(df_m)
            save_month(df_m, YEAR, m, out_dir)
        except Exception as e:
            print(f"❌ Error integrando {m}: {e}")

    # >>> CORRECCIÓN CLAVE: concatenar meses (apilar filas), NO hacer merge <<<
    if monthly_tables:
        df_all = pd.concat(monthly_tables, ignore_index=True)
        # Ordenar por tiempo y eliminar filas duplicadas de Timestamp (si algún mes se solapa)
        df_all = df_all.sort_values("Timestamp")
        df_all = df_all.drop_duplicates(subset=["Timestamp"], keep="first").reset_index(drop=True)

        # Guardar combinado mar–jul (o el rango que haya)
        has_mar = "marzo" in MONTHS
        has_jul = "julio" in MONTHS
        if has_mar and has_jul:
            rng_text = f"{YEAR}-03_07"
        else:
            rng_text = f"{YEAR}-{MONTH_TO_MM[MONTHS[0]]}_{MONTH_TO_MM[MONTHS[-1]]}"
        out_all = out_dir / f"SIMAT_BJU_{rng_text}.csv"
        df_all.to_csv(out_all, index=False)
        print(f"\n🧩 Combinado guardado: {out_all.name}")

        print("\nPreview combinado:")
        print(df_all.head(12))
    else:
        print("\nNo se generó tabla combinada.")
