import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, expon, norm, binom, poisson

# Función para generar valores usando transformada inversa
def generar_uniforme(a, b, size):
    u = np.random.uniform(0, 1, size)
    return a + (b - a) * u

def generar_exponencial(lambda_, size):
    u = np.random.uniform(0, 1, size)
    return -np.log(1 - u) / lambda_

def generar_normal(mu, sigma, size):
    u1 = np.random.uniform(0, 1, size)
    u2 = np.random.uniform(0, 1, size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mu + sigma * z0

# Parámetros y tamaño de muestra
size = 10000
a, b = 0, 1
mu, sigma = 0, 1
n, p = 100, 0.5
lambda1_ = 1
lambda2_ = 10
valores = [1, 2, 3, 4, 5]
probabilidades = [0.1, 0.2, 0.3, 0.25, 0.15]

# Generar valores
uniforme_valores = generar_uniforme(a, b, size)
exponencial_valores = generar_exponencial(lambda1_, size)
normal_valores = generar_normal(mu, sigma, size)
binomial_valores = np.random.binomial(n, p, size)
poisson_valores = np.random.poisson(lambda2_, size)
empirica_valores = np.random.choice(valores, size=size, p=probabilidades)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Histograma para la distribución binomial
axs[0, 0].hist(binomial_valores, bins=np.arange(0, n+2) - 0.5, density=True, alpha=0.6, color='g', label='Datos Binomial')
x = np.arange(0, n+1)
axs[0, 0].plot(x, binom.pmf(x, n, p), 'r-', label='Esperado')
axs[0, 0].set_title('Distribucion Binomial')
axs[0, 0].legend()

# Histograma para la distribución Poisson
axs[0, 1].hist(poisson_valores, bins=np.arange(0, np.max(poisson_valores)+2) - 0.5, density=True, alpha=0.6, color='g', label='Datos Poisson')
x = np.arange(0, np.max(poisson_valores)+1)
axs[0, 1].plot(x, poisson.pmf(x, lambda2_), 'r-', label='Esperado')
axs[0, 1].set_title('Distribucion Poisson')
axs[0, 1].legend()

# Histograma para la distribución empírica discreta
axs[0, 2].hist(empirica_valores, bins=np.arange(0.5, max(valores)+1.5), density=True, alpha=0.6, color='g', label='Datos Empirica')
axs[0, 2].plot(valores, probabilidades, 'ro', label='Esperado')
axs[0, 2].set_title('Distribucion Empirica Discreta')
axs[0, 2].legend()

# Histograma para la distribución uniforme
uniform_valores = np.random.uniform(0, 1, size)
axs[1, 0].hist(uniforme_valores, bins=50, density=True, alpha=0.6, color='g', label='Datos Uniforme')
axs[1, 0].plot(np.linspace(0, 1, 1000), [1]*1000, 'r-', label='Esperado')
axs[1, 0].set_title('Distribucion Uniforme')
axs[1, 0].legend()

# Histograma para la distribución exponencial
exponential_valores = np.random.exponential(1, size)
axs[1, 1].hist(exponencial_valores, bins=50, density=True, alpha=0.6, color='g', label='Datos Exponencial')
x = np.linspace(0, 10, 1000)
axs[1, 1].plot(x, expon.pdf(x), 'r-', label='Esperado')
axs[1, 1].set_title('Distribucion Exponencial')
axs[1, 1].legend()

# Histograma para la distribución normal
normal_valores = np.random.normal(0, 1, size)
axs[1, 2].hist(normal_valores, bins=50, density=True, alpha=0.6, color='g', label='Datos Normal')
x = np.linspace(-4, 4, 1000)
axs[1, 2].plot(x, norm.pdf(x), 'r-', label='Esperado')
axs[1, 2].set_title('Distribucion Normal')
axs[1, 2].legend()

plt.tight_layout()
plt.show()

'''
Generación de Valores:

Uniforme: Generamos valores uniformes en [a,b]utilizando la transformada inversa.
Exponencial: Generamos valores exponenciales con parámetro 𝜆 λ utilizando la transformada inversa.
Normal: Utilizamos el método de Box-Muller para generar valores normales con media 𝜇 y desviación estándar σ.

Histograma y Prueba de Bondad de Ajuste:

Se generan histogramas para visualizar la distribución de los datos generados.
Sobre los histogramas, se superpone la función de densidad de probabilidad teórica para la comparación visual.
Se realiza una prueba de Kolmogorov-Smirnov (K-S) para evaluar la bondad de ajuste entre los datos generados y la distribución teórica.
Uso del Código:
La prueba de K-S proporciona una medida estadística para evaluar la bondad de ajuste, complementando la inspección visual de los histogramas.
'''