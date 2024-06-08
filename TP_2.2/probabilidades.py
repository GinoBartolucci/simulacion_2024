import matplotlib.pyplot as plt
import math

# Parámetros del GCL
a = 1664525
c = 1013904223
m = 2**32
seed = 42

####Generador de números pseudoaleatorios con el GCL####
def gcl(seed, tamaño):
    numeros = []
    X = seed
    for _ in range(tamaño):
        X = (a * X + c) % m
        numeros.append(X / m)  # Normalizar a [0, 1)
    return numeros

####Distribución uniforme####
# Parámetros para la distribución uniforme
a_uniforme, b_uniforme = 0, 10  # Intervalo [a, b]
tamaño_uniforme = 10000  # Número de muestras

# Generación de la distribución uniforme
datos_uniforme = [a_uniforme+ (b_uniforme - a_uniforme) * x for x in gcl(seed, tamaño_uniforme)]

####Distribución normal####
# Parámetros para la distribución normal
mu, sigma = 0, 1  # Media y desviación estándar
tamaño_normal = 10000  # Número de muestras

# Generación de la distribución normal usando el método Box-Muller
def generar_normal(mu, sigma, tamaño, seed):
    ej_uniforme = gcl(seed, tamaño * 2)  # Necesitamos el doble de muestras uniformes
    datos_normal = []
    for i in range(0, len(ej_uniforme), 2):
        u1, u2 = ej_uniforme[i], ej_uniforme[i+1]
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        datos_normal.append(mu + z0 * sigma)
        if len(datos_normal) < tamaño:
            datos_normal.append(mu + z1 * sigma)
    return datos_normal

datos_normal = generar_normal(mu, sigma, tamaño_normal, seed)


