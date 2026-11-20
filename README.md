# Poromecánica Relacional Universal (U4) y Reología de Datos
Implementación computacional en Python de referencia correspondiente al manuscrito depositado en Zenodo.

## Estructura del Repositorio
* `modelo_core.py`: Constantes invariantes, operador de orden fraccionario variable $\beta(x)$, impedancia térmica de Boltzmann y construcción de matrices dispersas en formato CSR.
* `benchmarks_multiescala.py`: Reproducción estadística ($\chi^2_\nu$) y gráfica de los 7 casos de estudio multiescala de la Sección 5.
* `deducciones_ciegas.py`: Deducción analítica hacia adelante sin parámetros libres de cinco constantes empíricas (Kleiber, Reynolds, Feigenbaum, Von Kármán y ADN-B) correspondiente a la Subsección 5.2.
* `reologia_datos.py`: Algoritmo libre de parámetros para la detección ciega y reconstrucción de transiciones abruptas en vacíos de muestreo (Sección 6).

## Requisitos
`numpy`, `scipy`, `matplotlib`
