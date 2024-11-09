import load_dataset
from knn import knn
from evaluator import evaluator

train, train_labels, test, test_labels = load_dataset.load_iris_dataset(80)

knn = knn(k=3)
knn.fit(train, train_labels)

predictions = [knn.predict(point) for point in test]

accuracy = evaluator.accuracy(predictions, test_labels)

print("Jeux de test:")
for row in test_labels:
    print(row)

print("-----------------------------------------")

print("Predictions:")
for row in predictions:
    print(row)

print("-----------------------------------------")

print("Accuracy:")
print(accuracy)
