"""
SMAAcomp – Baseline Calibration Workflow for Low-Cost Air Quality Sensors
Version: 1.0 (Gold Theme)
Date: November 2025

Description:
------------
Baseline compensation and two-point calibration for low-cost AQ sensors using
reference station data (UNAM, CCA, IBERO). This edition renders every plot
in an elegant GOLD theme on TRANSPARENT background (for poster compositing).

Outputs (gold/transparent):
---------------------------
- correlation_before_compensation_gold.png
- correlation_after_compensation_gold.png
- correlation_after_comp_and_cal_gold.png
- time_series_comparison_gold.png
- ozone_comparison_gold.png
- error_analysis_temp_gold.png
- error_analysis_rh_gold.png
"""

# =============================================================================
# Metadata
# =============================================================================

__title__ = "SMAAcomp – Baseline Calibration Workflow for Low-Cost Air Quality Sensors"
__version__ = "1.0"
__date__ = "2025-11-11"
__authors__ = [
    "Horacio Serafín Jiménez Soto (Smability)",
    "Octavio Serafín Jiménez Soto (Smability)",
    "D. A. Pérez-De La Mora (Instituto de Investigación Aplicada y Tecnología, INIAT – Universidad Iberoamericana, Ciudad de México)"
]
__contact__ = {"Smability": "contacto@smability.com", "INIAT": "daniel.perez@ibero.mx"}
__license__ = "MIT"
__doi__ = "10.5281/zenodo.17583886"
__keywords__ = [
    "air quality","low-cost sensors","calibration","machine learning","ozone",
    "carbon monoxide","environmental monitoring","open science","Mexico City"
]
__repository__ = "https://github.com/SmabilityAI/SMAAcomp"
__description__ = (
    "Baseline workflow for compensating temperature and humidity effects and performing "
    "two-point calibration of low-cost air quality sensors using reference data from UNAM, "
    "CCA, and IBERO. Gold theme plots with transparent background for publishing."
)

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# =============================================================================
# GOLD THEME (transparent)
# =============================================================================
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

# =============================================================================
# Data I/O
# =============================================================================
def load_and_preprocess_data(cca_file, ruoa_file, ibero_file):
    # CCA Station
    df_cca = pd.read_csv(cca_file)
    df_cca['Fecha'] = pd.to_datetime(df_cca['Fecha']) + pd.to_timedelta(df_cca.Hora, unit='h')
    df_cca['ccao3'] = df_cca['ccao3'].interpolate()

    # RUOA Station
    df_ruoa = pd.read_csv(ruoa_file)
    df_ruoa['TIMESTAMP'] = pd.to_datetime(df_ruoa['TIMESTAMP'], format='mixed')
    df_ruoa_ = df_ruoa.set_index('TIMESTAMP').resample('60T').mean()
    df_ruoa_['RH_Avg'] = df_ruoa_['RH_Avg'].interpolate()
    df_ruoa_['Temp_Avg'] = df_ruoa_['Temp_Avg'].interpolate()

    # IBERO2
    df_ibero = pd.read_csv(ibero_file)  # ibero2 (UTC offset +1h)
    df_ibero['Timestamp'] = pd.to_datetime(df_ibero['Timestamp'], dayfirst=True) + pd.Timedelta(hours=1)
    df_ibero_ = df_ibero.set_index('Timestamp').resample('60T').mean()
    df_ibero_ = df_ibero_.loc['2023-03-07 15:01:00':'2023-04-04 14:47:00']
    df_ibero_['Ozone_Data'] = df_ibero_['Ozone_Data'].interpolate()
    df_ibero_['Temperature_Data'] = df_ibero_['Temperature_Data'].interpolate()
    df_ibero_['Relative_Humidity_Data'] = df_ibero_['Relative_Humidity_Data'].interpolate()

    return df_cca, df_ruoa_, df_ibero_

# =============================================================================
# Plots (gold)
# =============================================================================
def _decorate_legend(ax):
    leg = ax.legend(facecolor="none", edgecolor=GOLD)
    for t in leg.get_texts():
        t.set_color(TXT)

def plot_data_comparisons(df_cca, df_ruoa, df_ibero):
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), dpi=150)
    axs = axs.ravel()

    # 1) Temp
    axs[0].plot(df_ruoa.index, df_ruoa['Temp_Avg'], label='RUOA', color=GOLD)
    axs[0].plot(df_ibero.index, df_ibero['Temperature_Data'], label='IBERO', color=GOLD_LIGHT)
    axs[0].set_title('Temperature Comparison'); axs[0].set_xlabel('Time'); axs[0].set_ylabel('Temperature')
    axs[0].grid(True, linestyle=':', linewidth=1); _decorate_legend(axs[0])

    # 2) RH
    axs[1].plot(df_ruoa.index, df_ruoa['RH_Avg'], label='RUOA', color=GOLD)
    axs[1].plot(df_ibero.index, df_ibero['Relative_Humidity_Data'], label='IBERO', color=GOLD_LIGHT)
    axs[1].set_title('Relative Humidity Comparison'); axs[1].set_xlabel('Time'); axs[1].set_ylabel('Relative Humidity')
    axs[1].grid(True, linestyle=':', linewidth=1); _decorate_legend(axs[1])

    # 3) O3
    axs[2].plot(df_cca['Fecha'], df_cca['ccao3'], label='CCA', color=GOLD)
    axs[2].plot(df_ibero.index, df_ibero['Ozone_Data'], label='IBERO', color=GOLD_LIGHT)
    axs[2].set_title('Ozone Concentration Comparison'); axs[2].set_xlabel('Time'); axs[2].set_ylabel('Ozone Concentration')
    axs[2].grid(True, linestyle=':', linewidth=1); _decorate_legend(axs[2])

    plt.tight_layout()
    plt.savefig("data_comparisons_gold.png", dpi=300)
    plt.show()

def compensate_ozone_data(ibero_ozone, temp, humidity):
    out = []
    for o3, t, h in zip(ibero_ozone, temp, humidity):
        compensated = (o3/1.451) + (0.1034*t**2 - 2.225*t + 0.01223*h**2 - 1.984*h + 79)
        out.append(compensated)
    return out

def two_point_calibration(raw_data, reference_data):
    raw_low, raw_high = min(raw_data), max(raw_data)
    ref_low, ref_high = min(reference_data), max(reference_data)
    print("Perform two-point calibration. Raw Low/High are values after compensation")
    print("RawLow:", raw_low, "RawHigh:", raw_high, "RawRange:", (raw_high - raw_low))
    print("ReferenceLow:", ref_low, "ReferenceHigh:", ref_high, "ReferenceRange:", (ref_high - ref_low))
    return [((x - raw_low) * (ref_high - ref_low) / (raw_high - raw_low)) + ref_low for x in raw_data]

def calculate_error_metrics(reference, predicted):
    ref = np.array(reference); pred = np.array(predicted)
    abs_err = np.abs(ref - pred)
    mae = np.mean(abs_err)
    mape = np.mean(abs_err / np.where(ref==0, np.nan, ref)) * 100
    rmse = np.sqrt(np.mean(abs_err**2))
    return mae, mape, rmse

def plot_final_ozone_comparison(df_cca, cca_o3, compensated_o3, calibrated_o3):
    fig, ax = plt.subplots(figsize=(12,6), dpi=150)
    ax.plot(df_cca['Fecha'], cca_o3, label='CCA O3', color=GOLD)
    ax.plot(df_cca['Fecha'], compensated_o3, label='Compensated O3', color=GOLD_LIGHT)
    ax.plot(df_cca['Fecha'], calibrated_o3, label='Calibrated O3', color=GOLD_DIM)
    ax.set_title('Ozone Concentration Comparison'); ax.set_xlabel('Time'); ax.set_ylabel('Ozone Concentration')
    ax.grid(True, linestyle=':', linewidth=1); _decorate_legend(ax)
    plt.tight_layout(); plt.savefig("ozone_comparison_gold.png", dpi=300); plt.show()

def plot_correlation(reference, measured, title, filename=None):
    slope, intercept, r_value, p_value, _ = stats.linregress(reference, measured)

    fig, ax = plt.subplots(figsize=(10,6), dpi=150)
    ax.scatter(reference, measured, s=28, color=GOLD_LIGHT, alpha=0.55, edgecolor=GOLD, linewidth=0.3)

    mn = min(min(reference), min(measured)); mx = max(max(reference), max(measured))
    ax.plot([mn, mx], [mn, mx], linestyle="--", color=GOLD_DIM, linewidth=2, label="Perfect Correlation")
    ax.plot(reference, slope*np.array(reference)+intercept, color=GOLD, linewidth=2.4, label="Linear Regression")

    ax.set_xlabel('CCA Ozone Reference'); ax.set_ylabel('Ibero Ozone'); ax.set_title(title)
    eq = f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}\np-value = {p_value:.1e}'
    ax.text(0.02, 0.98, eq, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="none", edgecolor=GOLD, linewidth=1.2))
    ax.grid(True, linestyle=":", linewidth=1); _decorate_legend(ax)
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=300)
    plt.show()

def plot_time_series_comparison(df_cca, cca_o3, ibero_o3, compensated_o3, calibrated_o3):
    fig, axs = plt.subplots(2, 1, figsize=(12,10), dpi=150)

    axs[0].plot(df_cca['Fecha'], cca_o3, label='CCA O3', color=GOLD)
    axs[0].plot(df_cca['Fecha'], ibero_o3, label='Original IBERO O3', color=GOLD_LIGHT)
    axs[0].set_title('Original IBERO O3 vs CCA Reference'); axs[0].set_xlabel('Time'); axs[0].set_ylabel('Ozone Concentration')
    axs[0].grid(True, linestyle=':', linewidth=1); _decorate_legend(axs[0])

    axs[1].plot(df_cca['Fecha'], cca_o3, label='CCA O3', color=GOLD)
    axs[1].plot(df_cca['Fecha'], compensated_o3, label='Compensated', color=GOLD_LIGHT)
    axs[1].plot(df_cca['Fecha'], calibrated_o3, label='Comp. + Calibrated', color=GOLD_DIM)
    axs[1].set_title('Processed IBERO O3 vs CCA Reference'); axs[1].set_xlabel('Time'); axs[1].set_ylabel('Ozone Concentration')
    axs[1].grid(True, linestyle=':', linewidth=1); _decorate_legend(axs[1])

    plt.tight_layout(); plt.savefig("time_series_comparison_gold.png", dpi=300); plt.show()

# =============================================================================
# Error analysis (Temp / RH) – gold theme
# =============================================================================
def error_analysis_and_visualization(cca_O3, Ibero2_O3, tmpsma_avg, rhsma_avg):
    cca_O3 = np.array(cca_O3); Ibero2_O3 = np.array(Ibero2_O3)
    tmpsma_avg = np.array(tmpsma_avg); rhsma_avg = np.array(rhsma_avg)

    error_ppb = cca_O3 - Ibero2_O3
    error_percentage = 100 * (cca_O3 - Ibero2_O3) / np.where(cca_O3==0, np.nan, cca_O3)

    # ----- Temperature Figure -----
    fig, axs = plt.subplots(2, 2, figsize=(16,10), dpi=150)

    # Abs error vs Temp
    axs[0,0].scatter(tmpsma_avg, error_ppb, alpha=0.55, color=GOLD_LIGHT, edgecolor=GOLD, linewidth=0.3)
    m_temp_ppb, b_temp_ppb = np.polyfit(tmpsma_avg, error_ppb, 1)
    axs[0,0].plot(np.sort(tmpsma_avg), m_temp_ppb*np.sort(tmpsma_avg)+b_temp_ppb, color=GOLD, linewidth=2)
    axs[0,0].set_title('Absolute Error (ppb) vs Temperature'); axs[0,0].set_xlabel('Temperature'); axs[0,0].set_ylabel('Error (ppb)')
    axs[0,0].grid(True, linestyle=':', linewidth=1)
    axs[0,0].text(0.02,0.98,f'y = {m_temp_ppb:.4f}x {b_temp_ppb:+.4f}', transform=axs[0,0].transAxes,
                  ha='left', va='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    # % error vs Temp
    axs[0,1].scatter(tmpsma_avg, error_percentage, alpha=0.55, color=GOLD_LIGHT, edgecolor=GOLD, linewidth=0.3)
    m_temp_pct, b_temp_pct = np.polyfit(tmpsma_avg, error_percentage, 1)
    axs[0,1].plot(np.sort(tmpsma_avg), m_temp_pct*np.sort(tmpsma_avg)+b_temp_pct, color=GOLD, linewidth=2)
    axs[0,1].set_title('Percentage Error vs Temperature'); axs[0,1].set_xlabel('Temperature'); axs[0,1].set_ylabel('Error (%)')
    axs[0,1].grid(True, linestyle=':', linewidth=1)
    axs[0,1].text(0.02,0.98,f'y = {m_temp_pct:.4f}x {b_temp_pct:+.4f}', transform=axs[0,1].transAxes,
                  ha='left', va='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    # Quadratic abs
    axs[1,0].scatter(tmpsma_avg, error_ppb, color=GOLD_LIGHT, alpha=0.55, edgecolor=GOLD, linewidth=0.3)
    temp_ppb_coeffs = np.polyfit(tmpsma_avg, error_ppb, 2); temp_ppb_poly = np.poly1d(temp_ppb_coeffs)
    xx = np.sort(tmpsma_avg); axs[1,0].plot(xx, temp_ppb_poly(xx), color=GOLD, linewidth=2)
    axs[1,0].set_title('Quadratic Absolute Error (ppb) vs Temperature'); axs[1,0].set_xlabel('Temperature'); axs[1,0].set_ylabel('Error (ppb)')
    axs[1,0].grid(True, linestyle=':', linewidth=1)
    axs[1,0].text(0.02,0.98,f'y = {temp_ppb_coeffs[0]:.4f}x² {temp_ppb_coeffs[1]:+.4f}x {temp_ppb_coeffs[2]:+.4f}',
                  transform=axs[1,0].transAxes, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    # Quadratic %
    axs[1,1].scatter(tmpsma_avg, error_percentage, color=GOLD_LIGHT, alpha=0.55, edgecolor=GOLD, linewidth=0.3)
    temp_pct_coeffs = np.polyfit(tmpsma_avg, error_percentage, 2); temp_pct_poly = np.poly1d(temp_pct_coeffs)
    axs[1,1].plot(xx, temp_pct_poly(xx), color=GOLD, linewidth=2)
    axs[1,1].set_title('Quadratic Percentage Error vs Temperature'); axs[1,1].set_xlabel('Temperature'); axs[1,1].set_ylabel('Error (%)')
    axs[1,1].grid(True, linestyle=':', linewidth=1)
    axs[1,1].text(0.02,0.98,f'y = {temp_pct_coeffs[0]:.4f}x² {temp_pct_coeffs[1]:+.4f}x {temp_pct_coeffs[2]:+.4f}',
                  transform=axs[1,1].transAxes, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    plt.tight_layout(); plt.savefig("error_analysis_temp_gold.png", dpi=300); plt.show()

    # ----- RH Figure -----
    fig, axs = plt.subplots(2, 2, figsize=(16,10), dpi=150)

    axs[0,0].scatter(rhsma_avg, error_ppb, alpha=0.55, color=GOLD_LIGHT, edgecolor=GOLD, linewidth=0.3)
    m_rh_ppb, b_rh_ppb = np.polyfit(rhsma_avg, error_ppb, 1)
    xs = np.sort(rhsma_avg); axs[0,0].plot(xs, m_rh_ppb*xs+b_rh_ppb, color=GOLD, linewidth=2)
    axs[0,0].set_title('Absolute Error (ppb) vs Relative Humidity'); axs[0,0].set_xlabel('Relative Humidity'); axs[0,0].set_ylabel('Error (ppb)')
    axs[0,0].grid(True, linestyle=':', linewidth=1)
    axs[0,0].text(0.02,0.98,f'y = {m_rh_ppb:.4f}x {b_rh_ppb:+.4f}', transform=axs[0,0].transAxes,
                  ha='left', va='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    axs[0,1].scatter(rhsma_avg, error_percentage, alpha=0.55, color=GOLD_LIGHT, edgecolor=GOLD, linewidth=0.3)
    m_rh_pct, b_rh_pct = np.polyfit(rhsma_avg, error_percentage, 1)
    axs[0,1].plot(xs, m_rh_pct*xs+b_rh_pct, color=GOLD, linewidth=2)
    axs[0,1].set_title('Percentage Error vs Relative Humidity'); axs[0,1].set_xlabel('Relative Humidity'); axs[0,1].set_ylabel('Error (%)')
    axs[0,1].grid(True, linestyle=':', linewidth=1)
    axs[0,1].text(0.02,0.98,f'y = {m_rh_pct:.4f}x {b_rh_pct:+.4f}', transform=axs[0,1].transAxes,
                  ha='left', va='top', bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    axs[1,0].scatter(rhsma_avg, error_ppb, color=GOLD_LIGHT, alpha=0.55, edgecolor=GOLD, linewidth=0.3)
    rh_ppb_coeffs = np.polyfit(rhsma_avg, error_ppb, 2); rh_ppb_poly = np.poly1d(rh_ppb_coeffs)
    axs[1,0].plot(xs, rh_ppb_poly(xs), color=GOLD, linewidth=2)
    axs[1,0].set_title('Quadratic Absolute Error (ppb) vs Relative Humidity'); axs[1,0].set_xlabel('Relative Humidity'); axs[1,0].set_ylabel('Error (ppb)')
    axs[1,0].grid(True, linestyle=':', linewidth=1)
    axs[1,0].text(0.02,0.98,f'y = {rh_ppb_coeffs[0]:.4f}x² {rh_ppb_coeffs[1]:+.4f}x {rh_ppb_coeffs[2]:+.4f}',
                  transform=axs[1,0].transAxes, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    axs[1,1].scatter(rhsma_avg, error_percentage, color=GOLD_LIGHT, alpha=0.55, edgecolor=GOLD, linewidth=0.3)
    rh_pct_coeffs = np.polyfit(rhsma_avg, error_percentage, 2); rh_pct_poly = np.poly1d(rh_pct_coeffs)
    axs[1,1].plot(xs, rh_pct_poly(xs), color=GOLD, linewidth=2)
    axs[1,1].set_title('Quadratic Percentage Error vs Relative Humidity'); axs[1,1].set_xlabel('Relative Humidity'); axs[1,1].set_ylabel('Error (%)')
    axs[1,1].grid(True, linestyle=':', linewidth=1)
    axs[1,1].text(0.02,0.98,f'y = {rh_pct_coeffs[0]:.4f}x² {rh_pct_coeffs[1]:+.4f}x {rh_pct_coeffs[2]:+.4f}',
                  transform=axs[1,1].transAxes, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='none', edgecolor=GOLD))

    plt.tight_layout(); plt.savefig("error_analysis_rh_gold.png", dpi=300); plt.show()

    print("Error Statistics:")
    print(f"Mean Absolute Error (ppb): {np.mean(np.abs(error_ppb)):.4f}")
    print(f"Mean Percentage Error (%): {np.nanmean(np.abs(error_percentage)):.4f}")

    return {
        'temp_linear_ppb_coef': (m_temp_ppb, b_temp_ppb),
        'temp_linear_pct_coef': (m_temp_pct, b_temp_pct),
        'temp_quad_ppb_coef': temp_ppb_coeffs,
        'temp_quad_pct_coef': temp_pct_coeffs,
        'rh_linear_ppb_coef': (m_rh_ppb, b_rh_ppb),
        'rh_linear_pct_coef': (m_rh_pct, b_rh_pct),
        'rh_quad_ppb_coef': rh_ppb_coeffs,
        'rh_quad_pct_coef': rh_pct_coeffs
    }

# =============================================================================
# Main
# =============================================================================
def main():
    df_cca, df_ruoa, df_ibero = load_and_preprocess_data(
        'cca_o3_co_2023-04-11.csv',
        '2023-03-unam_hora_L1.csv',
        'ibero2_0703_0403.csv'
    )

    cca_o3 = df_cca['ccao3'].values
    ibero_o3 = df_ibero['Ozone_Data'].values
    ibero_temp = df_ibero['Temperature_Data'].values
    ibero_humidity = df_ibero['Relative_Humidity_Data'].values

    print("Initial Correlation:"); print(np.corrcoef(cca_o3, ibero_o3))

    compensated_o3 = compensate_ozone_data(ibero_o3, ibero_temp, ibero_humidity)

    error_analysis_results = error_analysis_and_visualization(
        cca_O3=cca_o3, Ibero2_O3=ibero_o3,
        tmpsma_avg=ibero_temp, rhsma_avg=ibero_humidity
    )

    calibrated_o3 = two_point_calibration(compensated_o3, cca_o3)

    print("\nBefore Compensation:")
    mae, mape, rmse = calculate_error_metrics(cca_o3, ibero_o3)
    print(f"MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}")

    print("\nAfter Compensation:")
    mae, mape, rmse = calculate_error_metrics(cca_o3, compensated_o3)
    print(f"MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}")

    print("\nAfter Calibration & Compensation:")
    mae, mape, rmse = calculate_error_metrics(cca_o3, calibrated_o3)
    print(f"MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}")

    print("\nPre-final Correlation Compensation:")
    print(stats.pearsonr(cca_o3, compensated_o3))

    print("\nFinal Correlation Calibration & Compensation:")
    print(stats.pearsonr(cca_o3, calibrated_o3))

    # Plots
    plot_data_comparisons(df_cca, df_ruoa, df_ibero)
    plot_final_ozone_comparison(df_cca, cca_o3, compensated_o3, calibrated_o3)
    plot_time_series_comparison(df_cca, cca_o3, ibero_o3, compensated_o3, calibrated_o3)

    plot_correlation(cca_o3, ibero_o3,
                     'Correlation Before Compensation',
                     'correlation_before_compensation_gold.png')

    plot_correlation(cca_o3, compensated_o3,
                     'Correlation After Compensation',
                     'correlation_after_compensation_gold.png')

    plot_correlation(cca_o3, calibrated_o3,
                     'Correlation After Compensation & Calibration',
                     'correlation_after_comp_and_cal_gold.png')

# =============================================================================
# Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"{__title__}"); print("=" * 80)
    print(f"Version: {__version__} | Date: {__date__}")
    print(f"Authors: {', '.join(__authors__)}")
    print(f"License: {__license__}")
    print(f"Repository: {__repository__}")
    print("-" * 80); print("Initializing baseline calibration workflow...\n")
    main()
    print("\nProcess completed successfully.")
    print("Generated outputs (gold/transparent):")
    print(" - data_comparisons_gold.png")
    print(" - ozone_comparison_gold.png")
    print(" - time_series_comparison_gold.png")
    print(" - correlation_before_compensation_gold.png")
    print(" - correlation_after_compensation_gold.png")
    print(" - correlation_after_comp_and_cal_gold.png")
    print(" - error_analysis_temp_gold.png")
    print(" - error_analysis_rh_gold.png")
    print("=" * 80)
