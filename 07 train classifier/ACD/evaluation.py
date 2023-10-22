from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from scipy.special import expit
import constants


def compute_metrics_ACD(eval_pred):
    predictions, lab = eval_pred
    predictions = (expit(predictions) > 0.5)
    labels = [l == 1 for l in lab]
    accuracy = accuracy_score(labels, predictions)

    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    hamming = hamming_loss(labels, predictions)

    metrics = {
        "hamming_loss": hamming,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
    }

    for i, aspect_category in enumerate(constants.ASPECT_CATEGORIES):
        class_labels = [label[i] for label in labels]
        class_predictions = [prediction[i] for prediction in predictions]

        precision = precision_score(class_labels, class_predictions, zero_division=0)
        recall = recall_score(class_labels, class_predictions, zero_division=0)
        f1 = f1_score(class_labels, class_predictions, zero_division=0)

        metrics[f"precision_{aspect_category}"] = precision
        metrics[f"recall_{aspect_category}"] = recall
        metrics[f"f1_{aspect_category}"] = f1

    return metrics
