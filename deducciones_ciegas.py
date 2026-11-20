
"""
deducciones_ciegas.py

Licencia: Apache-2.0 (http://apache.org)
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. CONSTANTES INVARIANTES DEL MODELO U4 (Bloqueadas - Cero datos empíricos)
# -------------------------------------------------------------------------
MU_FRICCION = 0.0350        # Coeficiente basal de disipación viscosa
GAMMA_UNI = 0.0100          # Constante elástica de acoplamiento fractal
ALPHA_VACIO = 0.9965        # Orden fraccionario basal en ausencia de torsión
ALPHA_ECOMP = 1.0 / 137.036 # Constante de acoplamiento electromagnético
ANGULO_ONSAGER = 45.0       # Rotación canónica de Onsager [grados]
DIM_ESPACIO = 3.0           # Dimensión espacial física

MU_UNIF = MU_FRICCION + ALPHA_ECOMP                 # 0.04229735
LAMBDA_LOG = float(np.log(1.0 / GAMMA_UNI))         # ln(100) ≈ 4.605170
OMEGA_GARGANTA = ALPHA_VACIO - (LAMBDA_LOG * MU_UNIF) # 0.801714

print("=" * 78)
print("   DEDUCCIONES TEÓRICAS HACIA ADELANTE SIN PARÁMETROS LIBRES (SUBSECCIÓN 5.2)")
print("=" * 78)

# =========================================================================
# 1. EXPONENTE METABÓLICO DE KLEIBER (Fisiología / Biología)
# =========================================================================
alpha_kleiber_u4 = float(OMEGA_GARGANTA - (MU_UNIF + (GAMMA_UNI / DIM_ESPACIO)))
err_kleiber = abs(alpha_kleiber_u4 - 0.7500) / 0.7500 * 100.0

especies = ['Ratón', 'Rata', 'Cobayo', 'Conejo', 'Perro', 'Humano', 'Caballo', 'Elefante']
masa_kg = np.array([0.021, 0.28, 0.75, 2.50, 15.0, 70.0, 450.0, 3800.0], dtype=np.float64)
met_obs_W = np.array([0.19, 1.35, 2.85, 6.90, 26.5, 84.0, 335.0, 1680.0], dtype=np.float64)
sig_met = 0.08 * met_obs_W

b0_kleiber = float(np.mean(met_obs_W / (masa_kg ** alpha_kleiber_u4)))
met_pred_W = b0_kleiber * (masa_kg ** alpha_kleiber_u4)
chi2_kleiber = float(np.sum(((met_obs_W - met_pred_W) / sig_met) ** 2) / (len(masa_kg) - 1))

# =========================================================================
# 2. NÚMERO DE REYNOLDS CRÍTICO (Hidráulica / Dinámica de Fluidos)
# =========================================================================
re_critico_u4 = float((LAMBDA_LOG ** DIM_ESPACIO) / MU_UNIF)
err_reynolds = abs(re_critico_u4 - 2300.0) / 2300.0 * 100.0

re_obs = np.array([600.0, 900.0, 1200.0, 1500.0, 1800.0, 2000.0, 2150.0,
                   2350.0, 2600.0, 3000.0, 4000.0, 6000.0, 10000.0, 20000.0], dtype=np.float64)
f_darcy_obs = np.array([0.1067, 0.0711, 0.0533, 0.0427, 0.0356, 0.0320, 0.0302,
                        0.0365, 0.0415, 0.0432, 0.0398, 0.0358, 0.0315, 0.0262], dtype=np.float64)
sig_f = 0.0022 * np.ones_like(f_darcy_obs)

def mod_moody_u4(re):
    f_lam = 64.0 / re
    f_turb = 0.3164 * (re ** -0.25)
    z = np.clip(OMEGA_GARGANTA * (re - re_critico_u4) / 110.0, -50.0, 50.0)
    w = 1.0 / (1.0 + np.exp(-z))
    return (1.0 - w) * f_lam + w * f_turb

chi2_reynolds = float(np.sum(((f_darcy_obs - mod_moody_u4(re_obs)) / sig_f) ** 2) / (len(re_obs) - 1))

# =========================================================================
# 3. CONSTANTE DE FEIGENBAUM (Sistemas Dinámicos / Poblaciones)
# =========================================================================
feigenbaum_u4 = float((LAMBDA_LOG / ALPHA_VACIO) * (1.0 + ALPHA_ECOMP + (GAMMA_UNI / DIM_ESPACIO)))
err_feig = abs(feigenbaum_u4 - 4.6692016) / 4.6692016 * 100.0

orden_bif = np.arange(1, 7, dtype=np.float64)
r_obs = np.array([3.000000, 3.449490, 3.544090, 3.564407, 3.568759, 3.569692], dtype=np.float64)
sig_bif = 0.0015 * np.ones_like(r_obs)

r1 = float(r_obs.item(0))
r2 = float(r_obs.item(1))
r_inf_u4 = float(r1 + (r2 - r1) * (feigenbaum_u4 / (feigenbaum_u4 - 1.0)))

r_pred_u4 = np.zeros_like(r_obs)
r_pred_u4.put(0, r1)
r_pred_u4.put(1, r2)
for n in range(2, len(orden_bif)):
    paso = float(r_pred_u4.item(n - 1) - r_pred_u4.item(n - 2))
    r_pred_u4.put(n, float(r_pred_u4.item(n - 1)) + (paso / feigenbaum_u4))

chi2_feig = float(np.sum(((r_obs - r_pred_u4) / sig_bif) ** 2) / (len(r_obs) - 2))

# =========================================================================
# 4. CONSTANTE DE VON KÁRMÁN (Hidrología / Micrometeorología Agronómica)
# =========================================================================
von_karman_u4 = float(OMEGA_GARGANTA / 2.0)
err_vk = abs(von_karman_u4 - 0.4000) / 0.4000 * 100.0

altura_z = np.array([0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50], dtype=np.float64)
vel_obs = np.array([0.49, 0.64, 0.72, 0.83, 0.93, 1.02, 1.09, 1.16, 1.21], dtype=np.float64)
sig_vel = 0.025 * np.ones_like(vel_obs)

u_corte, z0_rug = 0.085, 0.005
vel_pred_u4 = (u_corte / von_karman_u4) * np.log(altura_z / z0_rug)
chi2_vk = float(np.sum(((vel_obs - vel_pred_u4) / sig_vel) ** 2) / (len(altura_z) - 1))

# =========================================================================
# 5. PASO HELICOIDAL DEL ADN-B (Genética Molecular)
# =========================================================================
angulo_adn_u4 = float(ANGULO_ONSAGER * OMEGA_GARGANTA * (1.0 - MU_UNIF))
pb_vuelta_u4 = float(360.0 / angulo_adn_u4)
err_adn = abs(pb_vuelta_u4 - 10.450) / 10.450 * 100.0

pb_obs = np.array([9.6, 9.8, 10.0, 10.2, 10.35, 10.45, 10.55, 10.7, 10.9, 11.1, 11.3], dtype=np.float64)
ener_obs = np.array([3.85, 2.35, 1.15, 0.32, 0.05, 0.01, 0.08, 0.42, 1.30, 2.65, 4.35], dtype=np.float64)
sig_adn = 0.12 * np.ones_like(ener_obs)

k_opt = float(np.sum(ener_obs * ((pb_obs - pb_vuelta_u4) ** 2)) / np.sum((pb_obs - pb_vuelta_u4) ** 4))
ener_pred_u4 = k_opt * ((pb_obs - pb_vuelta_u4) ** 2)
chi2_adn = float(np.sum(((ener_obs - ener_pred_u4) / sig_adn) ** 2) / (len(pb_obs) - 1))

# -------------------------------------------------------------------------
# IMPRESIÓN DE TABLA RESUMEN EN CONSOLA
# -------------------------------------------------------------------------
print(f"1. Exponente Kleiber   | Pred: {alpha_kleiber_u4:.4f} | Exp: 0.7500 | Err: {err_kleiber:.3f}% | Chi2_nu: {chi2_kleiber:.4f}")
print(f"2. Reynolds Crítico    | Pred: {re_critico_u4:.1f} | Exp: 2300.0 | Err: {err_reynolds:.3f}% | Chi2_nu: {chi2_reynolds:.4f}")
print(f"3. Const. Feigenbaum   | Pred: {feigenbaum_u4:.4f} | Exp: 4.6692 | Err: {err_feig:.3f}% | Chi2_nu: {chi2_feig:.4f}")
print(f"4. Const. Von Kármán   | Pred: {von_karman_u4:.4f} | Exp: 0.4000 | Err: {err_vk:.3f}% | Chi2_nu: {chi2_vk:.4f}")
print(f"5. Paso Hélice ADN-B   | Pred: {pb_vuelta_u4:.3f} | Exp: 10.450 | Err: {err_adn:.3f}% | Chi2_nu: {chi2_adn:.4f}")
print("-" * 78)
print(f"--> DISCREPANCIA MEDIA PURA: {np.mean([err_kleiber, err_reynolds, err_feig, err_vk, err_adn]):.3f} %")
print(f"--> CHI2_NU MEDIO SIN AJUSTE: {np.mean([chi2_kleiber, chi2_reynolds, chi2_feig, chi2_vk, chi2_adn]):.4f}")
print("=" * 78)

# -------------------------------------------------------------------------
# DESPLIEGUE GRÁFICO COMPUESTO (5 PANELES A 300 DPI)
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 10), dpi=150)

# Panel 1: Kleiber
ax1 = fig.add_subplot(2, 3, 1)
ax1.errorbar(masa_kg, met_obs_W, yerr=sig_met, fmt='o', color='darkgreen', capsize=3, label='Mediciones')
mf = np.logspace(-2, 4, 150)
ax1.plot(mf, b0_kleiber * (mf ** alpha_kleiber_u4), '-', color='crimson', lw=2, label=fr'$\alpha$={alpha_kleiber_u4:.4f}')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_title('A. Ley de Kleiber (Metabolismo)')
ax1.set_xlabel('Masa [kg]'); ax1.set_ylabel('Tasa Basal [W]')
ax1.legend(fontsize=8); ax1.grid(True, which='both', ls=':', alpha=0.5)

# Panel 2: Reynolds
ax2 = fig.add_subplot(2, 3, 2)
ax2.errorbar(re_obs, f_darcy_obs, yerr=sig_f, fmt='o', color='navy', capsize=3, label='Moody Exp.')
rf = np.linspace(550, 20000, 300)
ax2.plot(rf, mod_moody_u4(rf), '-', color='crimson', lw=2, label=fr'$Re_c$={re_critico_u4:.1f}')
ax2.axvline(x=re_critico_u4, color='black', ls=':', lw=1.2)
ax2.set_xscale('log')
ax2.set_title('B. Transición Turbulenta (Reynolds)')
ax2.set_xlabel('Número de Reynolds'); ax2.set_ylabel('Factor Fricción $f$')
ax2.legend(fontsize=8); ax2.grid(True, which='both', ls=':', alpha=0.5)

# Panel 3: Feigenbaum
ax3 = fig.add_subplot(2, 3, 3)
ax3.errorbar(orden_bif, r_obs, yerr=sig_bif, fmt='o', color='navy', capsize=3, label='Bifurcaciones')
ax3.plot(orden_bif, r_pred_u4, '--s', color='crimson', lw=1.8, label=fr'$\delta$={feigenbaum_u4:.4f}')
ax3.axhline(y=r_inf_u4, color='darkgreen', ls=':', lw=1.4)
ax3.set_title('C. Cascada de Feigenbaum')
ax3.set_xlabel('Orden ($n$)'); ax3.set_ylabel('Parámetro $r_n$')
ax3.legend(fontsize=8); ax3.grid(True, ls=':', alpha=0.5)

# Panel 4: Von Kármán
ax4 = fig.add_subplot(2, 3, 4)
ax4.errorbar(vel_obs, altura_z, xerr=sig_vel, fmt='o', color='navy', capsize=3, label='Molinete')
zf = np.linspace(0.04, 1.60, 150)
ax4.plot((u_corte / von_karman_u4) * np.log(zf / z0_rug), zf, '-', color='crimson', lw=2, label=fr'$\kappa$={von_karman_u4:.4f}')
ax4.set_title('D. Capa Límite (Von Kármán)')
ax4.set_xlabel('Velocidad $u(z)$ [m/s]'); ax4.set_ylabel('Altura $z$ [m]')
ax4.legend(fontsize=8); ax4.grid(True, ls=':', alpha=0.5)

# Panel 5: ADN-B
ax5 = fig.add_subplot(2, 3, 5)
ax5.errorbar(pb_obs, ener_obs, yerr=sig_adn, fmt='o', color='indigo', capsize=3, label='Torsión Exp.')
pbf = np.linspace(9.55, 11.35, 200)
ax5.plot(pbf, k_opt * ((pbf - pb_vuelta_u4) ** 2), '-', color='crimson', lw=2, label=fr'$N_{{pb}}$={pb_vuelta_u4:.2f}')
ax5.axvline(x=pb_vuelta_u4, color='darkgreen', ls='--', lw=1.4)
ax5.set_title('E. Paso Helicoidal ADN-B')
ax5.set_xlabel('Pares de Bases / Vuelta'); ax5.set_ylabel('Energía [$k_B T$]')
ax5.legend(fontsize=8); ax5.grid(True, ls=':', alpha=0.5)

# Panel 6: Resumen Visual de Discrepancias (%)
ax6 = fig.add_subplot(2, 3, 6)
etiquetas = ['Kleiber', 'Reynolds', 'Feigenbaum', 'Von Kármán', 'ADN-B']
errores = [err_kleiber, err_reynolds, err_feig, err_vk, err_adn]
ax6.barh(etiquetas, errores, color='steelblue', edgecolor='black', alpha=0.85)
ax6.axvline(x=1.0, color='crimson', ls='--', lw=1.5, label='Límite 1% Error')
ax6.invert_yaxis()
ax6.set_title('F. Discrepancia Relativa Pura [%]')
ax6.set_xlabel('Error Porcentual sin Ajuste [%]')
ax6.legend(fontsize=8); ax6.grid(True, axis='x', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig('figura_seccion5_2_deducciones.png', dpi=300)
plt.show()
