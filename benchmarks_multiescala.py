
"""
benchmarks_multiescala.py
Validación empírica multi-escala (7 casos representativos de la Sección V).
Ejecutable de forma autónoma o en conjunto con modelo_core.py.
Licencia: Apache-2.0 (http://apache.org)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------------------------------------------------
# 1. CONSTANTES INVARIANTES DEL MODELO U4 (Bloqueadas)
# -------------------------------------------------------------------------
MU_FRICCION = 0.0350
GAMMA_UNI = 0.0100
ALPHA_VACIO = 0.9965
ALPHA_ECOMP = 1.0 / 137.036
BETA_POZO = 0.1700
K_BOLTZMANN = 1.380649e-23

MU_UNIF = MU_FRICCION + ALPHA_ECOMP
LAMBDA_LOG = float(np.log(1.0 / GAMMA_UNI))
OMEGA_GARGANTA = ALPHA_VACIO - (LAMBDA_LOG * MU_UNIF) # 0.801714

def beta_u4(x, x_c, gamma_loc, temp_K=293.15):
    z_b = 1.0 + (K_BOLTZMANN * temp_K / LAMBDA_LOG)
    s_eff = np.abs(x - x_c)
    beta = (ALPHA_VACIO / z_b) * ((1.0 + np.abs(gamma_loc) * (s_eff ** 2)) ** (-MU_UNIF))
    return np.clip(beta, BETA_POZO, ALPHA_VACIO)

def transicion_u4(x, y_0, delta_y, x_c, gamma_loc, temp_K=293.15, kappa=1.0):
    beta = beta_u4(x, x_c, gamma_loc, temp_K)
    exponente = np.clip((OMEGA_GARGANTA / beta) * kappa * (x - x_c), -100.0, 100.0)
    return y_0 + delta_y / (1.0 + np.exp(-exponente))

print("=" * 76)
print("   VALIDACIÓN EMPÍRICA MULTI-ESCALA: MODELO POROMECÁNICO U4 (SECCIÓN V)")
print("=" * 76)

# -------------------------------------------------------------------------
# CASO 1: ESCALA CUÁNTICA - TEST DE BELL (CHSH)
# -------------------------------------------------------------------------
angulos_deg = np.array([0.0, 11.25, 22.5, 28.59, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0], dtype=np.float64)
coincidencias_obs = np.array([0.98, 0.92, 0.78, 0.65, 0.54, 0.31, 0.14, 0.05, 0.01, 0.00], dtype=np.float64)
sig_bell = 0.025 * np.ones_like(coincidencias_obs)

def mod_bell(theta, y0, dy, tc, gam):
    return transicion_u4(theta, y0, dy, tc, gam, temp_K=2.725, kappa=-0.095)

popt_1, _ = curve_fit(mod_bell, angulos_deg, coincidencias_obs, p0=[0.0, 1.0, 32.0, 0.05],
                      bounds=([-0.05, 0.8, 20.0, 0.0001], [0.05, 1.2, 45.0, 5.0]), sigma=sig_bell, absolute_sigma=True)
chi2_1 = np.sum(((coincidencias_obs - mod_bell(angulos_deg, *popt_1)) / sig_bell) ** 2) / (len(angulos_deg) - 4)

# -------------------------------------------------------------------------
# CASO 2: ESCALA FLUIDODINÁMICA - DESCARGA COMPRESIBLE EN TOBERA
# -------------------------------------------------------------------------
x_tob = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.44, 0.47,
                  0.50, 0.54, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90], dtype=np.float64)
y_tob = np.array([0.18, 0.24, 0.31, 0.39, 0.48, 0.58, 0.69, 0.81, 0.91, 0.96,
                  0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], dtype=np.float64)
sig_tob = 0.02 * np.ones_like(y_tob)

def mod_tobera(x, y0, xc, gam):
    x_norm = np.maximum(0.0, x / xc)
    beta = beta_u4(x_norm ** 2, 0.0, gam, 298.15)
    flujo = y0 + (1.0 - y0) * (x_norm ** (2.0 * OMEGA_GARGANTA / beta))
    return np.minimum(1.0, flujo)

popt_2, _ = curve_fit(mod_tobera, x_tob, y_tob, p0=[0.15, 0.47, 0.2],
                      bounds=([0.05, 0.35, 0.001], [0.25, 0.60, 10.0]), sigma=sig_tob, absolute_sigma=True)
chi2_2 = np.sum(((y_tob - mod_tobera(x_tob, *popt_2)) / sig_tob) ** 2) / (len(y_tob) - 3)

# -------------------------------------------------------------------------
# CASO 3: ESCALA CRIOGÉNICA - HELIO-4 TRANSICIÓN LAMBDA
# -------------------------------------------------------------------------
t_he = np.array([1.50, 1.65, 1.80, 1.95, 2.05, 2.12, 2.15, 2.17, 2.19, 2.22,
                 2.28, 2.35, 2.45, 2.55, 2.60], dtype=np.float64)
cp_he = np.array([1.15, 1.85, 2.85, 4.30, 5.80, 7.60, 9.40, 11.80, 7.10, 4.20,
                  2.80, 2.35, 2.15, 2.08, 2.05], dtype=np.float64)
sig_he = 0.35 * np.ones_like(cp_he)

def mod_lambda(temp_K, cp_norm, pico_l, tc, af, ac, gam):
    beta = beta_u4(temp_K, tc, gam, np.maximum(0.1, temp_K))
    ancho = np.where(temp_K <= tc, af, ac)
    pico = pico_l * np.exp(-np.abs(temp_K - tc) / ancho) * (OMEGA_GARGANTA / beta)
    fondo = np.where(temp_K <= tc, 0.35 * ((temp_K / tc) ** 3), cp_norm)
    return fondo + pico

popt_3, _ = curve_fit(mod_lambda, t_he, cp_he, p0=[2.05, 11.0, 2.17, 0.22, 0.04, 1.0],
                      bounds=([1.20, 6.0, 2.10, 0.08, 0.01, 0.001], [2.80, 18.0, 2.22, 0.60, 0.15, 25.0]),
                      sigma=sig_he, absolute_sigma=True)
chi2_3 = np.sum(((cp_he - mod_lambda(t_he, *popt_3)) / sig_he) ** 2) / (len(t_he) - 6)

# -------------------------------------------------------------------------
# CASO 4: ESCALA GEOFÍSICA - EXTRAPOLACIÓN VOLCÁN CALBUCO (2015)
# -------------------------------------------------------------------------
h_calbuco = np.array([-36.0, -30.0, -24.0, -18.0, -14.0, -10.0, -8.0, -7.0], dtype=np.float64)
indice_vm = np.array([0.12, 0.14, 0.17, 0.22, 0.29, 0.42, 0.56, 0.68], dtype=np.float64)
sig_cal = 0.03 * np.ones_like(indice_vm)

def mod_calbuco(t_h, base, amp, t_rup, gam):
    dt = (t_h - t_rup) / 10.0
    beta = beta_u4(dt, 0.0, gam, 1150.0)
    return base + amp * np.exp(OMEGA_GARGANTA * (1.0 / beta) * dt * 1.1)

popt_4, _ = curve_fit(mod_calbuco, h_calbuco, indice_vm, p0=[0.10, 0.90, -1.5, 0.2],
                      bounds=([0.05, 0.50, -5.0, 0.001], [0.20, 1.50, 2.0, 10.0]), sigma=sig_cal, absolute_sigma=True)
chi2_4 = np.sum(((indice_vm - mod_calbuco(h_calbuco, *popt_4)) / sig_cal) ** 2) / (len(h_calbuco) - 4)

# -------------------------------------------------------------------------
# CASO 5: ESCALA HIDROLÓGICA - RÍO MENDOZA (NATURAL VS REGULADO DGI)
# -------------------------------------------------------------------------
x_rio = np.arange(1, 13, dtype=np.float64)
y_nat = np.array([22.5, 21.8, 24.0, 32.5, 58.0, 96.5, 118.0, 94.0, 56.0, 35.0, 26.5, 23.5], dtype=np.float64)
y_reg = np.array([26.0, 31.5, 39.0, 48.5, 62.0, 74.0, 78.5, 71.0, 54.0, 41.0, 32.0, 27.5], dtype=np.float64)
sig_rio = 3.5 * np.ones_like(x_rio)

def mod_rio(t, q0, dq, tc, tau, gam):
    dt = (t - tc) / tau
    beta = beta_u4(dt, 0.0, gam, 288.15)
    return q0 + dq * np.exp(-0.5 * (dt ** 2) * (OMEGA_GARGANTA / beta))

popt_5a, _ = curve_fit(mod_rio, x_rio, y_nat, p0=[22.0, 95.0, 7.0, 1.5, 0.5],
                       bounds=([15, 50, 5.5, 0.5, 0.001], [35, 140, 8.5, 4.0, 15]), sigma=sig_rio, absolute_sigma=True)
popt_5b, _ = curve_fit(mod_rio, x_rio, y_reg, p0=[25.0, 55.0, 6.8, 2.5, 0.1],
                       bounds=([15, 30, 5.5, 1.0, 0.001], [40, 100, 8.5, 5.5, 15]), sigma=sig_rio, absolute_sigma=True)
chi2_5a = np.sum(((y_nat - mod_rio(x_rio, *popt_5a)) / sig_rio) ** 2) / (12 - 5)
chi2_5b = np.sum(((y_reg - mod_rio(x_rio, *popt_5b)) / sig_rio) ** 2) / (12 - 5)

# -------------------------------------------------------------------------
# CASO 6: ESCALA FISIOLÓGICA - CICATRIZACIÓN DÉRMICA
# -------------------------------------------------------------------------
x_dia = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                  10.0, 11.0, 12.0, 14.0, 16.0, 18.0], dtype=np.float64)
y_her = np.array([0.0, 2.5, 6.0, 12.5, 22.0, 35.5, 51.0, 66.0, 78.5, 87.0,
                  92.5, 95.8, 97.6, 99.1, 99.7, 99.9], dtype=np.float64)
sig_her = 2.0 * np.ones_like(y_her)

def mod_herida(dia, y0, dy, xc, gam):
    return transicion_u4(dia, y0, dy, xc, gam, temp_K=310.15, kappa=0.72)

popt_6, _ = curve_fit(mod_herida, x_dia, y_her, p0=[0.0, 100.0, 6.0, 0.15],
                      bounds=([-5, 85, 3.0, 0.0001], [5, 105, 10.0, 10]), sigma=sig_her, absolute_sigma=True)
chi2_6 = np.sum(((y_her - mod_herida(x_dia, *popt_6)) / sig_her) ** 2) / (len(y_her) - 4)

# -------------------------------------------------------------------------
# CASO 7: ESCALA ECOLÓGICA - FRAGMENTACIÓN Y PERCOLACIÓN DEL PAISAJE
# -------------------------------------------------------------------------
x_hab = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                  55.0, 60.0, 65.0, 70.0, 80.0, 90.0, 100.0], dtype=np.float64)
y_con = np.array([1.5, 3.0, 5.5, 9.5, 16.0, 27.5, 44.0, 63.5, 79.0, 88.5,
                  93.8, 96.5, 98.0, 98.9, 99.5, 99.8, 100.0], dtype=np.float64)
sig_eco = 2.5 * np.ones_like(y_con)

def mod_paisaje(hab, y0, dy, xc, gam):
    return transicion_u4(hab / 10.0, y0, dy, xc / 10.0, gam, temp_K=293.15, kappa=1.45)

popt_7, _ = curve_fit(mod_paisaje, x_hab, y_con, p0=[0.0, 100.0, 36.0, 0.2],
                      bounds=([-5, 85, 20.0, 0.0001], [5, 105, 55.0, 10]), sigma=sig_eco, absolute_sigma=True)
chi2_7 = np.sum(((y_con - mod_paisaje(x_hab, *popt_7)) / sig_eco) ** 2) / (len(y_con) - 4)

# -------------------------------------------------------------------------
# IMPRESIÓN DE TABLA CONSOLIDADA EN TERMINAL
# -------------------------------------------------------------------------
chi2_lista = [chi2_1, chi2_2, chi2_3, chi2_4, chi2_5a, chi2_5b, chi2_6, chi2_7]
print(f"1. Test de Bell CHSH       | Chi2_nu = {chi2_1:.4f} | tc       = {float(popt_1[2]):.2f} deg")
print(f"2. Tobera Supersónica      | Chi2_nu = {chi2_2:.4f} | P_ext/P0 = {1.0 - float(popt_2[1]):.4f}")
print(f"3. Helio-4 Punto Lambda    | Chi2_nu = {chi2_3:.4f} | T_lambda = {float(popt_3[2]):.4f} K")
print(f"4. Volcán Calbuco (2015)   | Chi2_nu = {chi2_4:.4f} | t_rup    = {float(popt_4[2]):.2f} h")
print(f"5a. Río Mendoza (Natural)  | Chi2_nu = {chi2_5a:.4f} | tau      = {float(popt_5a[3]):.2f} meses")
print(f"5b. Río Mendoza (Regulado) | Chi2_nu = {chi2_5b:.4f} | tau      = {float(popt_5b[3]):.2f} meses")
print(f"6. Cicatrización Dérmica   | Chi2_nu = {chi2_6:.4f} | T50      = {float(popt_6[2]):.2f} días")
print(f"7. Fragmentación Paisaje   | Chi2_nu = {chi2_7:.4f} | H50      = {float(popt_7[2]):.2f} %")
print("-" * 76)
print(f"--> MEDIA GLOBAL DE BONDAD DE AJUSTE (8 SERIES): {float(np.mean(chi2_lista)):.4f}")
print("=" * 76)

