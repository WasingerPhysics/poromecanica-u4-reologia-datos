
"""
modelo_core.py
Núcleo matemático de la Poromecánica Relacional Universal sobre Variedades U4.
Referencia: Secciones II, III y IV del manuscrito en Zenodo.
Licencia: Apache-2.0 (http://apache.org)
"""

import numpy as np
from scipy.sparse import csr_matrix

# -------------------------------------------------------------------------
# 1. CONSTANTES ESTRUCTURALES INVARIANTES (Secciones 2.2 y 3.1)
# -------------------------------------------------------------------------
MU_FRICCION = 0.0350        # Coeficiente basal de disipación viscosa
GAMMA_UNI = 0.0100          # Constante elástica de acoplamiento fractal
ALPHA_VACIO = 0.9965        # Orden fraccionario basal en ausencia de torsión
ALPHA_ECOMP = 1.0 / 137.036 # Constante de acoplamiento electromagnético (estructura fina)
GAMMA_ROBIN = 1.5000        # Factor de contención elástica en frontera
BETA_POZO = 0.1700          # Límite inferior subdifusivo extremo
K_BOLTZMANN = 1.380649e-23  # Constante de Boltzmann [J/K]

MU_UNIF = MU_FRICCION + ALPHA_ECOMP                 # 0.04229735
LAMBDA_LOG = float(np.log(1.0 / GAMMA_UNI))         # ln(100) ≈ 4.605170
OMEGA_GARGANTA = ALPHA_VACIO - (LAMBDA_LOG * MU_UNIF) # Invariante Garganta Coaxial: 0.801714

def impedancia_boltzmann(temp_K):
    """Calcula el factor adimensional de impedancia térmica Z_B(T) (Sección 2.3)."""
    return 1.0 + (K_BOLTZMANN * temp_K / LAMBDA_LOG)

def calcular_beta_variable(x, x_c, gamma_loc, temp_K=293.15):
    """
    Calcula el exponente fraccionario variable beta(x) acotado en [0.1700, 0.9965]
    acoplando torsión efectiva S(x) = |x - x_c| e impedancia térmica (Sección 2.3).
    """
    z_b = impedancia_boltzmann(temp_K)
    s_eff = np.abs(x - x_c)
    base_torsion = 1.0 + np.abs(gamma_loc) * (s_eff ** 2)
    beta = (ALPHA_VACIO / z_b) * (base_torsion ** (-MU_UNIF))
    return np.clip(beta, BETA_POZO, ALPHA_VACIO)

def ecuacion_transicion_u4(x, y_0, delta_y, x_c, gamma_loc, temp_K=293.15, kappa=1.0):
    """
    Ecuación general de transición de régimen modulada por Omega_garganta (0.801714).
    """
    beta = calcular_beta_variable(x, x_c, gamma_loc, temp_K)
    exponente = np.clip((OMEGA_GARGANTA / beta) * kappa * (x - x_c), -100.0, 100.0)
    return y_0 + delta_y / (1.0 + np.exp(-exponente))

def construir_matriz_csr(distancias_norm, beta_nodos, r_poro=0.25, temp_K=293.15):
    """
    Construye el Tensor de Adyacencia y Permeabilidad Variable en formato CSR
    anulando conexiones superiores al radio crítico r_poro (Sección 4.1).
    """
    n_nodos = len(beta_nodos)
    filas, columnas, valores = [], [], []
    denom_termico = 1.0 + K_BOLTZMANN * temp_K

    for i in range(n_nodos):
        for j in range(n_nodos):
            if i != j and distancias_norm[i, j] <= r_poro:
                d_ij = distancias_norm[i, j]
                alpha_ij = 0.5 * (beta_nodos[i] + beta_nodos[j]) * (
                    (1.0 + (GAMMA_ROBIN * (d_ij ** 2)) / denom_termico) ** (-MU_FRICCION)
                )
                l_ij = 1.0 / ((d_ij + 1e-4) ** alpha_ij)
                filas.append(i)
                columnas.append(j)
                valores.append(l_ij)

    return csr_matrix((valores, (filas, columnas)), shape=(n_nodos, n_nodos))

if __name__ == "__main__":
    print(f"Núcleo U4 cargado correctamente | Omega_garganta = {OMEGA_GARGANTA:.6f}")
