# Import libraries
import numpy as np
import pandas as pd
import random

#---------------
# Función de distancia euclideana
# La función calcula la distancia euclideana, 
# la cuál es la longitud del segmento de línea que conecta los puntos. 
# La distancia euclideana también se conoce como la norma L2 de un vector.
#---------------

def euclideanDistance(a, b):
    return np.sum((a - b)**2)

#-----------------
# Función de KMeans
# Parámetros de la función
# dataset: Conjunto de datos en 2D.
# k: Número de clusteres.
# is_kmeans: De forma predeterminada, el algoritmo utilizará 
# la variable booleana TRUE para usar el KMeans estandar
# de lo contrario, utilizará el algoritmo de KMeans++.
# is_random: La muestra más larga del conjunto de datos
# se selecciona como el nuevo centro de clúster cuando se utiliza KMeans
# de lo contrario, se inicializará de manera aleatoria 
# en función de la probabilidad de distancia.
#-----------------

def _KMeans_(dataset, k, is_kmeans=True, is_random=False):
    # Definiendo las dimensiones de las variables
    num_sample, num_feature = dataset.shape
    # K-means++: inicializaremos el centro de agrupación en clústeres
    if not is_kmeans:
        # Seleccionamos de manera inicial
        # el primer punto central
        first_index = random.sample(range(num_sample), 1)
        center = dataset[first_index]
        # Calcule la distancia de cada muestra desde cada punto central y
        # seleccionaremos otros puntos centrales a través del método de la ruleta
        dist_note = np.zeros(num_sample)
        dist_note += 1000000000.0

        for j in range(k):
            if j+1 == k:
                # En este punto
                # Se han calculado suficientes centros de agrupación en
                # clústeres para salir directamente.
                break

            # Para este punto de ejecución
            # Se calcularán las distancias entre cada muestra y el centro
            # del clúster, guardando la distancia más pequeña.
            for i in range(num_sample):
                dist = euclideanDistance(center[j], dataset[i])
                if dist < dist_note[i]:
                    dist_note[i] = dist

            # De este modo, si se utiliza el método de la ruleta,
            # se genera aleatoriamente un nuevo centro de clúster
            # en función de la distancia, de lo contrario se utiliza
            # la muestra más distante como el siguiente punto central del clúster.
            if is_random:
                dist_p = dist_note / dist_note.sum()
                next_idx = np.random.choice(range(num_sample), 1, p=dist_p)
                center = np.vstack([center, dataset[next_idx]])
            else:
                next_idx = dist_note.argmax()
                center = np.vstack([center, dataset[next_idx]])

    # K-Means: Aqui inicializamos aleatoriamente los centros de clúster.
    else:
        # Inicializamos de manera aleatoria el punto
        # central del clúster.
        center_indexs = random.sample(range(num_sample), k)
        center = dataset[center_indexs, :]

    # Implementación iterativa del KMeans
    cluster_assessment = np.zeros((num_sample, 2))
    # Todas las features las estableceremos
    # con un valor -1
    cluster_assessment[:, 0] = -1
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False

        for i in range(num_sample):
            min_distance = 100000000.0
            c = 0
            # A este punto determinaremos a qué clase pertenece cada muestra,
            # es decir, qué punto central está más cerca de ella.
            for j in range(k):
                dist = euclideanDistance(dataset[i, :], center[j, :])
                if min_distance > dist:
                    min_distance = dist
                    c = j
            # Paso de actualización de la asignación del clúster
            # Todavía hay cambios de categoría en los datos en los dos cálculos anteriores
            # que no cumplen el requisito de detención de iteración.
            if cluster_assessment[i, 0] != c:
                cluster_assessment[i, :] = c, min_distance
                cluster_changed = True
        # Paso de actualización para la posición
        # del punto central del clúster.
        for j in range(k):
            changed_center = dataset[cluster_assessment[:, 0] == j].mean(
                axis=0)
            center[j, :] = changed_center
    return cluster_assessment, center
