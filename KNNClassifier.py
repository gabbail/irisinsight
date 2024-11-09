import math


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, data, labels):
        self.train_data = data
        self.train_labels = labels

    def predict(self, data_point):
        distances = []

        for i, value in enumerate(self.train_data):
            label = self.train_labels[i].item()
            distance = self.euclidian_distance(data_point, value)
            couple = (distance, label)
            distances.append(couple)


        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]

        vote_counts = {}

        for distance,label in neighbors:
            if label not in vote_counts:
                vote_counts[label] = 1
            else:
                vote_counts[label] += 1

        label_most_found = max(zip(vote_counts.values(), vote_counts.keys()))[1]

        return label_most_found

    def euclidian_distance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
