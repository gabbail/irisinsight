import numpy
import random
import math


def load_iris_dataset(train_ratio):
    file = open("datasets/bezdekIris.data", "r")
    random.seed(1)

    content = []

    for line in file.readlines():
        content.append(line.strip())

    random.shuffle(content)

    #conversion_labels = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    data = []
    labels = []

    n = len(content)

    for x in range(n):
        split = content[x].split(",")
        data.append(split[:4])
        labels.append(split[4])

    #numeric_labels = [conversion_labels[label] for label in labels]

    nbEntrainements = math.floor(n * train_ratio / 100)

    train = data[:nbEntrainements]
    test = data[nbEntrainements:]
    train_labels = labels[:nbEntrainements]
    test_labels = labels[nbEntrainements:]

    file.close()

    train = numpy.array(train, dtype=float)
    train_labels = numpy.array(train_labels)
    test = numpy.array(test, dtype=float)
    test_labels = numpy.array(test_labels)

    return train, train_labels, test, test_labels
