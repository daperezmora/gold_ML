import pandas as pd

df = pd.read_csv("SIMAT_BJU_2025-03_07.csv", parse_dates=["Timestamp"])
df = df.sort_values("Timestamp")

# % de disponibilidad por variable
for col in ["CO_ref","O3_ref","RH_ref","Temp_ref"]:
    if col in df.columns:
        pct = 100 * df[col].notna().mean()
        print(f"{col}: {pct:.1f}% de cobertura")

# Conteo por mes (útil para ver meses con más 'nr' -> NaN)
df["month"] = df["Timestamp"].dt.to_period("M").astype(str)
summary = df.groupby("month")[["CO_ref","O3_ref","RH_ref","Temp_ref"]].apply(lambda x: x.notna().mean().mul(100))
print("\nCobertura por mes (%)")
print(summary.round(1))
