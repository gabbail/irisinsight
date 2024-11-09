class evaluator:
    @staticmethod
    def accuracy(predictions, actual_labels):
        correct = 0
        total = len(actual_labels)
        for predicted, actual in zip(predictions, actual_labels):
            if predicted == actual:
                correct += 1
        return correct / total
