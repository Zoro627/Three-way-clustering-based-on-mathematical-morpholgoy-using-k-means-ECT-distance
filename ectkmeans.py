import math

import networkx as nx
import numpy as np
import pykov
import random

from configuration import K, EPS, LIMIT


def euclidean_distance(x, y):
    return math.sqrt(np.power(x - y, 2).sum()) + EPS


def get_k_nearest_neighbours(dataset):
    nearest_neighbours = []
    for i, x in enumerate(dataset):
        neighbours = []
        for j, y in enumerate(dataset):
            if i != j:
                neighbours.append((i, j, {'weight': euclidean_distance(x, y)}))

        neighbours.sort(key=lambda e: e[2]['weight'])
        nearest_neighbours.append(neighbours[:K])

    flat_list = [item for sublist in nearest_neighbours for item in sublist]
    return flat_list


def get_mst_edges(dataset):
    graph = nx.Graph()
    for i, x in enumerate(dataset):
        for j, y in enumerate(dataset):
            if i != j:
                graph.add_edge(i, j, weight=euclidean_distance(x, y))

    mst = nx.minimum_spanning_edges(graph, data=True)
    return list(mst)


def calculate_average_commute_time(average_first_passage_time_matrix):
    m = average_first_passage_time_matrix
    n = len(average_first_passage_time_matrix)
    ect_distance = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ect_distance[i][j] = math.sqrt(m[i][j] + m[j][i])

    return ect_distance


def calculate_average_first_passage_time(probability):
    state_status = {}
    n = len(probability)

    for i in range(n):
        for j in range(n):
            state_status[(i, j)] = probability[i][j]

    markov_chain = pykov.Chain(state_status)

    average_first_passage_time_matrix = np.zeros((n, n))
    for i in range(n):
        average_steps = markov_chain.mfpt_to(i)
        for j in average_steps:
            average_first_passage_time_matrix[i][j] = average_steps[j]

    return average_first_passage_time_matrix


def calculate_probability_matrix(adjacency_matrix):
    n = len(adjacency_matrix)
    probability = np.zeros((n, n))
    for i in range(n):
        a_i = np.sum(adjacency_matrix[i])
        for j in range(n):
            probability[i][j] = adjacency_matrix[i][j] / a_i

    return probability


def calculate_adjacency_matrix(graph):
    n = graph.number_of_nodes()
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and graph.has_edge(i, j):
                adjacency_matrix[i][j] = 1 / graph.get_edge_data(i, j)['weight']['weight']

    return adjacency_matrix


def make_base_graph(dataset):
    graph = nx.Graph()

    nearest_neighbours = get_k_nearest_neighbours(dataset)
    mst_edges = get_mst_edges(dataset)

    graph.add_weighted_edges_from(nearest_neighbours)
    graph.add_weighted_edges_from(mst_edges)

    return graph


def calculate_ect_distance(data):
    graph = make_base_graph(data)
    adjacency_matrix = calculate_adjacency_matrix(graph)
    probability = calculate_probability_matrix(adjacency_matrix)
    afpt_matrix = calculate_average_first_passage_time(probability)
    ect_distance = calculate_average_commute_time(afpt_matrix)
    return ect_distance


def ect_kmeans_clustering(ect_distance):
    n = len(ect_distance)
    prototype = random.sample(range(n), K)
    pre_criterion = 0

    while True:
        clusters = [[e] for e in prototype]
        for point in range(n):
            if point not in prototype:
                dist = []
                for c, p in enumerate(prototype):
                    dist.append((c, np.power(ect_distance[point][p], 2)))
                dist.sort(key=lambda e: e[1])
                clusters[dist[0][0]].append(point)

        for c in range(K):
            min_dist = []
            for candidate in clusters[c]:
                sum_dist = 0
                for other in clusters[c]:
                    if candidate != other:
                        sum_dist += np.power(ect_distance[candidate][other], 2)

                min_dist.append((candidate, sum_dist))

            min_dist.sort(key=lambda e: e[1])
            prototype[c] = min_dist[0][0]

        criterion = 0
        for c in range(K):
            for member in clusters[c]:
                if member != prototype[c]:
                    criterion += np.power(ect_distance[member][prototype[c]], 2)

        if pre_criterion == 0:
            pre_criterion = criterion

        elif criterion < pre_criterion:
            if pre_criterion - criterion < LIMIT:
                break
            else:
                pre_criterion = criterion
        else:
            break

    return clusters


def fit(data):
    predicts = np.zeros(len(data))
    ect_distance = calculate_ect_distance(data)
    clusters = ect_kmeans_clustering(ect_distance)
    for index, c in enumerate(clusters):
        for member in c:
            predicts[member] = index

    return predicts
