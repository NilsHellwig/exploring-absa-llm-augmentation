from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from scipy.special import expit
import constants


def compute_metrics_ACD(eval_pred):
    predictions, lab = eval_pred
    predictions = (expit(predictions) > 0.5)
    labels = [l == 1 for l in lab]
    accuracy = accuracy_score(labels, predictions)

    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_weighted = f1_score(labels, predictions, average="weighted")

    class_f1_scores = f1_score(labels, predictions, average=None)

    hamming = hamming_loss(labels, predictions)

    metrics = {
        "hamming_loss": hamming,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "class_f1_scores": class_f1_scores.tolist(),
    }

    for i, aspect_category in enumerate(constants.ASPECT_CATEGORIES):
        class_labels = [label[i] for label in labels]
        class_predictions = [prediction[i] for prediction in predictions]

        precision = precision_score(class_labels, class_predictions)
        recall = recall_score(class_labels, class_predictions)

        metrics[f"precision_{aspect_category}"] = precision
        metrics[f"recall_{aspect_category}"] = recall

    return metrics
