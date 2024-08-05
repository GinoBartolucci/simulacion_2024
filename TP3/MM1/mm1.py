import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
import pandas as pd
import os
from datetime import datetime
import argparse


def theoretical_values(arrival_rate, service_rate, max_queue_length):
    rho = arrival_rate / service_rate
    if rho >= 1:
        raise ValueError("El sistema es inestable para λ/μ ≥ 1")

    if max_queue_length == 0:
        # Si la capacidad de la cola es cero
        P_0 = 1 - rho
        P_denial = rho
        L = rho
        L_q = 0
        W_q = 0
        W = 1 / service_rate
    else:
        # Probabilidad de que un cliente se encuentre con cero clientes en cola
        P_0 = (1 - rho) / (1 - rho ** (max_queue_length + 2))
        # Probabilidad de que un cliente se encuentre con la cola llena
        P_denial = P_0 * rho ** (max_queue_length + 1)
        # Promedio de clientes en el sistema
        L = rho * (1 - (max_queue_length + 1) * rho ** max_queue_length + max_queue_length * rho ** (max_queue_length + 1)) / ((1 - rho) * (1 - rho ** (max_queue_length + 1)))
        # Promedio de clientes en cola
        L_q = L - rho
        # Tiempo promedio de espera
        W_q = L_q / arrival_rate
        # Tiempo promedio de espera en sistema
        W = W_q + (1 / service_rate)

    return {
        'rho': rho,
        'L': L,
        'L_q': L_q,
        'W': W,
        'W_q': W_q,
        'P_0': P_0,
        'P_denial': P_denial
    }


def simulate_mm1_queue(arrival_rate, service_rate, simulation_time, max_queue_length):
    t = 0
    n = 0  # Número de clientes en el sistema
    event_queue = []
    arrival_times = []
    departure_times = []
    area_n = 0
    last_event_time = 0
    denied_service_count = 0  # Contador de clientes denegados

    next_arrival = random.expovariate(arrival_rate)
    heapq.heappush(event_queue, (next_arrival, 'arrival'))

    while t < simulation_time:
        t, event_type = heapq.heappop(event_queue)
        area_n += n * (t - last_event_time)
        last_event_time = t

        if event_type == 'arrival':
            if n < max_queue_length + 1:
                arrival_times.append(t)
                if n == 0:
                    next_service = t + random.expovariate(service_rate)
                    heapq.heappush(event_queue, (next_service, 'departure'))
                n += 1
            else:
                denied_service_count += 1
            next_arrival = t + random.expovariate(arrival_rate)
            heapq.heappush(event_queue, (next_arrival, 'arrival'))
        elif event_type == 'departure':
            departure_times.append(t)
            n -= 1
            if n > 0:
                next_service = t + random.expovariate(service_rate)
                heapq.heappush(event_queue, (next_service, 'departure'))

    L = area_n / simulation_time
    W = sum([d - a for a, d in zip(arrival_times, departure_times)]) / len(arrival_times)
    L_q = L - (arrival_rate / service_rate)
    W_q = W - (1 / service_rate)
    rho = arrival_rate / service_rate
    P_0 = 1 - rho
    P_denial = denied_service_count / (denied_service_count + len(arrival_times))

    return {
        'L': L,
        'W': W,
        'L_q': L_q,
        'W_q': W_q,
        'rho': rho,
        'P_0': P_0,
        'P_denial': P_denial,
        'event_log': event_queue
    }


# Parámetros iniciales ingresados por el usuario

parser = argparse.ArgumentParser(description="Simulador de M/M/1")
parser.add_argument("-s", default=2024, type=int, help="Número de semilla")
parser.add_argument("-l", default=10.0, type=float, help="Número de arrivo")
parser.add_argument("-m", default=15.0, type=float, help="Número de servicio")
parser.add_argument("-t", default=10000, type=int, help="Tiempo de simulacion")
parser.add_argument("-n", default=10, type=int, help="Cantidad de corridas")
args = parser.parse_args()
seed_value = args.s
arrival_rate = args.l
service_rate = args.m
simulation_time = args.t
num_runs = args.n
random.seed(seed_value)
np.random.seed(seed_value)
arrival_rate_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25]
queue_lengths = [0, 2, 5, 10, 50]

results = {
    'theoretical': [],
    'simulated': []
}

# Run simulations and theoretical calculations
for multiplier in arrival_rate_multipliers:
    for max_queue_length in queue_lengths:
        adjusted_arrival_rate = arrival_rate * multiplier
        theoretical = theoretical_values(adjusted_arrival_rate, service_rate, max_queue_length)
        simulated_runs = [simulate_mm1_queue(adjusted_arrival_rate, service_rate, simulation_time, max_queue_length) for _ in range(num_runs)]

        avg_simulated = {
            'L': np.mean([run['L'] for run in simulated_runs]),
            'W': np.mean([run['W'] for run in simulated_runs]),
            'L_q': np.mean([run['L_q'] for run in simulated_runs]),
            'W_q': np.mean([run['W_q'] for run in simulated_runs]),
            'rho': np.mean([run['rho'] for run in simulated_runs]),
            'P_0': np.mean([run['P_0'] for run in simulated_runs]),
            'P_denial': np.mean([run['P_denial'] for run in simulated_runs])
        }

        results['theoretical'].append(theoretical)
        results['simulated'].append(avg_simulated)

# Crear un DataFrame para la tabla
data = []
for i, (theoretical, simulated) in enumerate(zip(results['theoretical'], results['simulated'])):
    multiplier = arrival_rate_multipliers[i // len(queue_lengths)]
    queue_length = queue_lengths[i % len(queue_lengths)]
    data.append([f"{multiplier}-{queue_length}"] +
                [theoretical['L'], simulated['L']] +
                [theoretical['W'], simulated['W']] +
                [theoretical['L_q'], simulated['L_q']] +
                [theoretical['W_q'], simulated['W_q']] +
                [theoretical['rho'], simulated['rho']] +
                [theoretical['P_0'], simulated['P_0']] +
                [theoretical['P_denial'], simulated['P_denial']])

columns = ['Multiplier-Queue Length', 'L Teorico', 'L Simulado', 'W Teorico', 'W Simulado',
           'L_q Teorico', 'L_q Simulado', 'W_q Teorico', 'W_q Simulado',
           'rho Teorico', 'rho Simulado', 'P_0 Teorico', 'P_0 Simulado',
           'P_denial Teorico', 'P_denial Simulado']
df = pd.DataFrame(data, columns=columns)

# Guardar el DataFrame en un archivo CSV
ruta_relativa_carpeta = os.path.join("datos_generados")

# Obtener la fecha y hora actual
ahora = datetime.now()

# Formatear la fecha y hora para incluir en el nombre del archivo
timestamp = ahora.strftime("%Y%m%d_%H%M%S")

# Crear el nombre del archivo con el timestamp
nombre_archivo = f"simulation_results_{timestamp}.csv"

csv_path = os.path.join(ruta_relativa_carpeta, nombre_archivo)
df.to_csv(csv_path, index=False)

# Imprimir la tabla de forma legible en la consola
print(df)


def plot_results(results, metric):
    theoretical_values = [res[metric] for res in results['theoretical']]
    simulated_values = [res[metric] for res in results['simulated']]
    multipliers = [f"{m}-{ql}" for m in arrival_rate_multipliers for ql in queue_lengths]

    plt.figure(figsize=(10, 6))
    plt.plot(multipliers, theoretical_values, label='Teórico', marker='o')
    plt.plot(multipliers, simulated_values, label='Simulado', marker='x')
    plt.xlabel('Multiplicador de Tasa de Arribo - Longitud de Cola')
    plt.ylabel(metric)
    plt.title(f'Comparación de {metric}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_filename = f"grafica_{metric}_{timestamp}.jpg"
    plot_path = os.path.join(ruta_relativa_carpeta, plot_filename)
    plt.savefig(plot_path)
    plt.close()


metrics = ['L', 'W', 'L_q', 'W_q', 'rho', 'P_0', 'P_denial']
for metric in metrics:
    plot_results(results, metric)
