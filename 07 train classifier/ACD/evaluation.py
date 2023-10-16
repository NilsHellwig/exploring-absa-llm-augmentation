from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from scipy.special import expit


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

    return metrics
