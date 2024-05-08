from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from scipy.special import expit
from helper import save_pred_and_labels
import constants


def compute_metrics_ACSA(results, cross_idx):
    def compute_metrics(eval_pred):
        predictions, lab = eval_pred
        save_pred_and_labels(predictions, lab, results, cross_idx)

        predictions = (expit(predictions) > 0.5)
        labels = [l == 1 for l in lab]
        accuracy = accuracy_score(labels, predictions)

        f1_macro = f1_score(labels, predictions,
                            average="macro", zero_division=0)
        f1_micro = f1_score(labels, predictions,
                            average="micro", zero_division=0)
        f1_weighted = f1_score(labels, predictions,
                               average="weighted", zero_division=0)

        hamming = hamming_loss(labels, predictions)

        metrics = {
            "hamming_loss": hamming,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
        }

        # performance for each category
        for i, aspect_category in enumerate(constants.ASPECT_CATEGORIES):
            class_labels = [label[i*3:i*3+3] for label in labels]
            class_predictions = [prediction[i*3:i*3+3]
                                 for prediction in predictions]

            precision = precision_score(
                class_labels, class_predictions, zero_division=0, average='micro')
            recall = recall_score(
                class_labels, class_predictions, zero_division=0, average='micro')
            f1 = f1_score(class_labels, class_predictions,
                          zero_division=0, average='micro')
            accuracy = accuracy_score(class_labels, class_predictions)

            metrics[f"precision_{aspect_category}"] = precision
            metrics[f"recall_{aspect_category}"] = recall
            metrics[f"f1_{aspect_category}"] = f1
            metrics[f"accuracy_{aspect_category}"] = accuracy
            metrics[f"n_examples_{aspect_category}"] = sum(
                1 in sublist for sublist in class_labels)

        # performance for each category and sentiment
        for i, aspect_category_sentiment in enumerate(constants.ASPECT_CATEGORY_POLARITIES):
            class_labels = [label[i] for label in labels]
            class_predictions = [prediction[i] for prediction in predictions]

            precision = precision_score(
                class_labels, class_predictions, zero_division=0)
            recall = recall_score(
                class_labels, class_predictions, zero_division=0)
            f1 = f1_score(class_labels, class_predictions, zero_division=0)
            accuracy = accuracy_score(class_labels, class_predictions)

            if all(el == 0 for el in class_labels):
                metrics[f"precision_{aspect_category_sentiment}"] = "no examples for evaluation"
                metrics[f"recall_{aspect_category_sentiment}"] = "no examples for evaluation"
                metrics[f"f1_{aspect_category_sentiment}"] = "no examples for evaluation"
                metrics[f"accuracy_{aspect_category_sentiment}"] = "no examples for evaluation"
                metrics[f"n_examples_{aspect_category_sentiment}"] = "no examples for evaluation"
            else:

                metrics[f"precision_{aspect_category_sentiment}"] = precision
                metrics[f"recall_{aspect_category_sentiment}"] = recall
                metrics[f"f1_{aspect_category_sentiment}"] = f1
                metrics[f"accuracy_{aspect_category_sentiment}"] = accuracy
                metrics[f"n_examples_{aspect_category_sentiment}"] = class_labels.count(
                    1)

        return metrics
    return compute_metrics
