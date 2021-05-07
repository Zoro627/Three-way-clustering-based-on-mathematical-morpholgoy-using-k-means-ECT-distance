import numpy as np
import pandas

from sklearn.datasets import make_circles, make_moons
from configuration import DATA


def load_dataset():
    dataset = {
        'Circles': make_circles_data(),
        'Moons': make_moons_data(),
        'GlassIdentification': load_glass_identification(),
        'Ionosphere': load_ionosphere_data(),
        'Iris': load_iris_data(),
        'Wine': load_wine_data()
    }

    print('Datasets Loaded Successfully!\n')
    return dataset


def load_glass_identification():
    data = np.array(pandas.read_csv(filepath_or_buffer=DATA.GLASS_IDENTIFICATION, sep=',', header=None))
    labels = [int(value) - 1 for value in data[:, len(data[0]) - 1]]
    return {'data': np.array(data[:, 1:len(data[0]) - 1]), 'labels': np.array(labels)}


def load_ionosphere_data():
    data = np.array(pandas.read_csv(filepath_or_buffer=DATA.IONOSPHERE, sep=',', header=None))
    target = dict([(y, x + 1) for x, y in enumerate(sorted(set(data[:, len(data[0]) - 1])))])
    labels = [int(target[x]) - 1 for x in data[:, len(data[0]) - 1]]
    return {'data': np.array(data[:, :len(data[0]) - 1]), 'labels': np.array(labels)}


def load_iris_data():
    data = np.array(pandas.read_csv(filepath_or_buffer=DATA.IRIS, sep=',', header=None))
    target = dict([(y, x + 1) for x, y in enumerate(sorted(set(data[:, len(data[0]) - 1])))])
    labels = [int(target[x]) - 1 for x in data[:, len(data[0]) - 1]]
    return {'data': np.array(data[:, :len(data[0]) - 1]), 'labels': np.array(labels)}


def load_wine_data():
    data = np.array(pandas.read_csv(filepath_or_buffer=DATA.WINE, sep=',', header=None))
    labels = [int(value) - 1 for value in data[:, 0]]
    return {'data': np.array(data[:, 1:]), 'labels': np.array(labels)}


def make_circles_data():
    data, target = make_circles(n_samples=400, shuffle=True, noise=0.05, factor=0.5)
    return {'data': np.array(data), 'labels': np.array(target)}


def make_moons_data():
    data, target = make_moons(n_samples=400, shuffle=True, noise=0.05)
    return {'data': np.array(data), 'labels': np.array(target)}

