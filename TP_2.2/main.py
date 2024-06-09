import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse
import time

parser = argparse.ArgumentParser(description="Simulador de ruleta")
parser.add_argument("-n", default=10000, type=int, help="Número de pruebas")
parser.add_argument("-s", default=1000, type=int, help="seed")
args = parser.parse_args()
largo_secuencia = args.n
seed = args.s

if largo_secuencia < 1:
    print("El número de pruebas debe ser mayor a 0")
    exit()
if largo_secuencia <= 5000:
    size_puntos_graficos = 15
if largo_secuencia > 5000 and largo_secuencia <= 20000:
    size_puntos_graficos = 10
if largo_secuencia > 20000:
    size_puntos_graficos = 3
seed = int(time.time()) # Usar el tiempo actual como semilla, las misma para los 2 test

################################RUTAS################################
# Obtener la ruta completa del directorio actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

ruta_graficas = os.path.join(directorio_actual, 'graficas/')
if not os.path.exists(ruta_graficas):
    os.makedirs(ruta_graficas)

ruta_generadas = os.path.join(ruta_graficas, 'generadas/')
if not os.path.exists(ruta_generadas):
    os.makedirs(ruta_generadas)

ruta_graficas_numpy = os.path.join(ruta_graficas, 'numpy/')
if not os.path.exists(ruta_graficas_numpy):
    os.makedirs(ruta_graficas_numpy)

ruta_graficas_comparacion = os.path.join(ruta_graficas, 'comparacion/')
if not os.path.exists(ruta_graficas_comparacion):
    os.makedirs(ruta_graficas_comparacion)

def dist_uniforme(a, b, size):
  u = np.random.uniform(0, 1, size)
  return a + (b - a) * u

# Definimos la distribución exponencial
def dist_exponencial(lambd, largo_secuencia_exponencial):
    # Generamos la secuencia de números pseudoaleatorios
    secuencia = np.random.uniform(0, 1, largo_secuencia_exponencial)
    # Generación de la distribución exponencial con metodo de inversion
    datos_exponencial = [-math.log(x) / lambd for x in secuencia]
    return datos_exponencial

def dist_gamma(k, a, size):
  datos_gamma = []
  for i in range(size):
    tr = 1
    for i in range(k):
      r = np.random.uniform(0, 1)
      tr = tr * r
    datos_gamma.append(-1/a * np.log(tr))
  return datos_gamma

def dist_normal(mu, sigma, size):
  datos_normal = []
  for i in range(size):
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    datos_normal.append(mu + sigma * z0)
  return datos_normal

#binomial negativa 
def dist_pascal(k, q, size): 
  datos_pascal = []
  for i in range(size):
    x=0
    tr=1
    qr= np.log(q)
    for i in range(k):
      r = np.random.uniform(0, 1)
      tr = tr * r
    nx = np.log(tr)/qr
    x = nx
    datos_pascal.append(x)
  return datos_pascal
    
def dist_binomial(n, p, size):
  datos_binomial = []
  for i in range(size):
    x = 0
    for i in range(n):
      r = np.random.uniform(0, 1)
      if r <= p:
        x = x + 1
    datos_binomial.append(x)
  return datos_binomial

def dist_hipergeometrica(tn, ns, p, size):
  datos_hipergeometrica = []
  for i in range(size):
    x = 0
    tnx = tn
    px = p
    for i in range(ns):
      r = np.random.uniform(0, 1)
      if r <= px:
        s=1
        x= x+1
      else:
        s=0
      px = (tnx*p-s)/(tnx-1)
      tnx = tnx-1
    datos_hipergeometrica.append(x)
  return datos_hipergeometrica

def dist_poisson(p, size):
  datos_poisson = []
  for i in range(size):
    x = 0
    b = np.exp(-p)
    tr = 1
    while True:
        r = np.random.uniform(0, 1)
        tr = tr * r
        if tr < b:
          break
        x = x + 1
    datos_poisson.append(x)
  return datos_poisson

def dist_empirica_discreta(p, size):
  datos_empirica_discreta = []
  for i in range(size):
    r = np.random.uniform(0, 1)
    x = 0
    F = p[0]
    while r > F:
      x = x + 1
      F = F + p[x]
    datos_empirica_discreta.append(x)
  return datos_empirica_discreta

def graficar_histograma(datos, nombre=""):
    plt.figure(figsize=(14, 8))  
    bins = 40
    color = 'royalblue'
    histograma = os.path.join(ruta_generadas, f'histograma_{nombre}.png')
    if "Numpy" in nombre:
      histograma = os.path.join(ruta_graficas_numpy, f'histograma_{nombre}.png')
      color = 'limegreen'
    if "comparacion" in nombre: 
      histograma = os.path.join(ruta_graficas_comparacion, f'histograma_{nombre}.png')
      color = ['limegreen', 'royalblue']
    if ("Pascal" or "Binomial" or "Hipergeometrica" or "Poisson" or "Empirica_Discreta" ) in nombre: 
      bins = np.arange(0, 21) - 0.5
     
    plt.hist(datos, bins=bins, edgecolor='black', color=color)
    plt.title(f'Histograma de Frecuencias: {nombre}. N={largo_secuencia}')
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.savefig(histograma)
    plt.close()

def graficar_dispersion(datos, nombre=""):
    if "Numpy" in nombre:
      dispersion = os.path.join(ruta_graficas_numpy, f'dispersion_{nombre}.png')
      color = 'limegreen'
    else:
      dispersion = os.path.join(ruta_generadas, f'dispersion_{nombre}.png')
      color = 'royalblue'
    plt.figure(figsize=(14, 8))
    plt.scatter(range(len(datos)), datos, label=nombre, color=color, s=size_puntos_graficos)
    plt.xlabel('Índice')
    plt.ylabel('Número Pseudoaleatorio')
    plt.title(f'Gráfico de Dispersión de Números Pseudoaleatorios: {nombre}. N={largo_secuencia}')
    plt.savefig(dispersion)
    plt.close()

def uniforme():
  sec_u = np.random.uniform(0, 1, largo_secuencia)
  datos_uniforme = dist_uniforme(0, 1, largo_secuencia)
  # Graficamos la distribución uniforme
  graficar_histograma(datos_uniforme, "Uniforme")
  graficar_dispersion(datos_uniforme, "Uniforme")
  graficar_histograma(sec_u, "Numpy-Uniforme")
  graficar_dispersion(sec_u, "Numpy-Uniforme")
  graficar_histograma([sec_u, datos_uniforme], "Uniforme-comparacion")
def exponencial():
  sec_e = np.random.exponential(3, largo_secuencia)
  datos_exponencial = dist_exponencial(1/3, largo_secuencia)
  # Graficamos la distribución exponencial
  graficar_histograma(datos_exponencial, "Exponencial")
  graficar_dispersion(datos_exponencial, "Exponencial")
  graficar_histograma(sec_e, "Numpy-Exponencial")
  graficar_dispersion(sec_e, "Numpy-Exponencial")
  graficar_histograma([sec_e, datos_exponencial], "Exponencial-comparacion")
def gamma():
  sec_g = np.random.gamma(5, 20, largo_secuencia)
  datos_gamma = dist_gamma(5, 1/20, largo_secuencia)
  # Graficamos la distribución gamma
  graficar_histograma(datos_gamma, "Gamma")
  graficar_dispersion(datos_gamma, "Gamma")
  graficar_histograma(sec_g, "Numpy-Gamma")
  graficar_dispersion(sec_g, "Numpy-Gamma")
  graficar_histograma([sec_g, datos_gamma], "Gamma-comparacion")
def normal():
  sec_n = np.random.normal(0, 1, largo_secuencia)
  datos_normal = dist_normal(0, 1, largo_secuencia)
  # Graficamos la distribución normal
  graficar_histograma(datos_normal, "Normal")
  graficar_dispersion(datos_normal, "Normal")
  graficar_histograma(sec_n, "Numpy-Normal")
  graficar_dispersion(sec_n, "Numpy-Normal")
  graficar_histograma([sec_n, datos_normal], "Normal-comparacion")
def pascal():
  sec_p = np.random.negative_binomial(5, 0.5, largo_secuencia)
  datos_pascal = dist_pascal(5, 0.5, largo_secuencia)
  graficar_histograma(datos_pascal, "Pascal")
  graficar_dispersion(datos_pascal, "Pascal")
  # Graficamos la distribución binomial
  graficar_histograma(sec_p, "Numpy-Pascal")
  graficar_dispersion(sec_p, "Numpy-Pascal")
  graficar_histograma([sec_p, datos_pascal], "Pascal-comparacion")
def binomial():
  #ensayos(x) = 10, probabilidad(p) = 0.5, tamaño = 10000
  sec_b = np.random.binomial(20, 0.1, largo_secuencia)
  datos_binomial = dist_binomial(20, 0.1, largo_secuencia)
  graficar_histograma(datos_binomial, "Binomial")
  graficar_dispersion(datos_binomial, "Binomial")  
  # Graficamos la distribución binomial
  graficar_histograma(sec_b, "Numpy-Binomial")
  graficar_dispersion(sec_b, "Numpy-Binomial")
  graficar_histograma([sec_b, datos_binomial], "Binomial-comparacion")
def hipergeometrica():
  sec_h = np.random.hypergeometric(100, 60, 10, largo_secuencia)
  datos_hipergeometrica = dist_hipergeometrica(100, 60, 10/100, largo_secuencia)
  graficar_histograma(datos_hipergeometrica, "Hipergeometrica")
  graficar_dispersion(datos_hipergeometrica, "Hipergeometrica")
  # Graficamos la distribución binomial
  graficar_histograma(sec_h, "Numpy-Hipergeometrica")
  graficar_dispersion(sec_h, "Numpy-Hipergeometrica")
  graficar_histograma([sec_h, datos_hipergeometrica], "Hipergeometrica-comparacion")
def poisson():
  sec_po = np.random.poisson(5, largo_secuencia)
  datos_poisson = dist_poisson(5, largo_secuencia)
  graficar_histograma(datos_poisson, "Poisson")
  graficar_dispersion(datos_poisson, "Poisson")
  # Graficamos la distribución binomial
  graficar_histograma(sec_po, "Numpy-Poisson")
  graficar_dispersion(sec_po, "Numpy-Poisson")
  graficar_histograma([sec_po, datos_poisson], "Poisson-comparacion")
def empirica_discreta():
  p = [0.1, 0.2, 0.3, 0.4]
  sec_ed = np.random.choice([0, 1, 2, 3], largo_secuencia, p=p)
  datos_empirica_discreta = dist_empirica_discreta(p, largo_secuencia)
  graficar_histograma(datos_empirica_discreta, "Empirica_Discreta")
  graficar_dispersion(datos_empirica_discreta, "Empirica_Discreta")
  # Graficamos la distribución binomial
  graficar_histograma(sec_ed, "Numpy-Empirica_Discreta")
  graficar_dispersion(sec_ed, "Numpy-Empirica_Discreta")
  graficar_histograma([sec_ed, datos_empirica_discreta], "Empirica_Discreta-comparacion")

if __name__ == "__main__":
  uniforme()
  exponencial()
  gamma()
  normal()
  pascal()
  binomial()
  hipergeometrica()
  poisson()
  empirica_discreta()
  print("Gráficos generados exitosamente")