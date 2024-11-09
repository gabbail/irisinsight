import load_dataset
from KNNClassifier import KNNClassifier

train, train_labels, test, test_labels = load_dataset.load_iris_dataset(80)

knn = KNNClassifier(k=3)
knn.fit(train, train_labels)

predictions = [knn.predict(point) for point in test]

print(predictions)
