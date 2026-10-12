## Simulaciones y Análisis del Modelo de Ising 2D.

Este repositorio consiste en algunas simulaciones del Modelo de Ising en dos dimensiones con el algoritmo de Metropolis en Julia. A continuación se presenta un desglose de lo que hace cada archivo.

### Algoritmo de Metropolis-Hastings y Modelo de Ising

- `metrpolis_algorithm.ipynb`. Este notebook contiene una explicación del algoritmo de Metrópolis aplicado al modelo de Ising, junto con una análisis de la magnetización, la energía, y el comportamiento del sistema a distintas temperaturas.
- `parallel_metropolis.jl`. Este script contiene una implementación de la misma simulación de `metropolis_algorithm.ipynb` pero usando paralelismo y otras optimizaciones. Sus resultados para el análisis de early warnings se pueden encontrar en `simulation_results/resultados_transicion_fase.png`.
- `get_energy_benchmark_test.ipynb`. Este notebook consiste en una comparación con benchmarks entre distintos algoritmos para encontrar la energía en un sistema de Ising 2D como el implementado en `metropolis_algorithm.ipynb`. 

### Finite Scaling Analysis y Exponentes Críticos

- `fss_simulation_metropolis.jl`. Este script cuenta con una simulación del modelo de Ising a través del algoritmo de **Metropolis** implementada con paralelismo que genera los datos necesarios para hacer una análisis con cumulante de Binder de los exponentes críticos. Estos datos son guardados en `simulation_results/_metropolis_binder_data.csv` y `simulation_results/metropolis_sim_data.csv`.
- `fss_simulation_wolff.jl`. Este script cuenta con una simulación del modelo de Ising a través del algoritmo de **Wolff** implementada con paralelismo que genera los datos necesarios para hacer una análisis con cumulante de Binder de los exponentes críticos. Estos datos son guardados en `simulation_results/wolff_binder_data.csv` y `simulation_results/wolff_sim_data.csv`.
- `binder_analysis_metropolis.jl`. Este notebook pretender analizar los datos obtenidos de `fss_simulation_metropolis.jl`, obtener los exponentes críticos y explicar sus resultados.
- `binder_analysis_wolff.jl`. Este notebook pretender analizar los datos obtenidos de `fss_simulation_wolff.jl`, obtener los exponentes críticos y explicar sus resultados.