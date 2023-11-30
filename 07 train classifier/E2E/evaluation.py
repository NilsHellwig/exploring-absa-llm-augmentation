from helper import divide, save_pred_and_labels
import numpy as np
import constants


def add_end_to_tag(tag, i):
    tag["end"] = i
    return tag


def compute_metrics_for_subset(examples_predictions, examples_labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(examples_predictions)):
        example_metrics = calculate_tp_tn_fp_fn_spans(
            examples_predictions[i], examples_labels[i])
        tp += example_metrics[0]
        tn += example_metrics[1]
        fp += example_metrics[2]
        fn += example_metrics[3]
    return tp, tn, fp, fn


def calculate_tp_tn_fp_fn_spans(pred, label):
    pred_set = set(
        f"{range['start']}_{range['end']}_{range['polarity']}" for range in pred)
    label_set = set(
        f"{range['start']}_{range['end']}_{range['polarity']}" for range in label)

    # Calculate true positives by finding the intersection of the sets.
    tp_set = pred_set & label_set
    tp = len(tp_set)

    # Calculate false positives by subtracting the intersection from the predicted set.
    fp_set = pred_set - tp_set
    fp = len(fp_set)

    # Calculate false negatives by subtracting the intersection from the label set.
    fn_set = label_set - tp_set
    fn = len(fn_set)

    return tp, 0, fp, fn


def get_predicted_phrases(example):
    total_tags = []
    for polarity in constants.POLARITIES:
        tags_for_polarity = []

        for i in range(len(example)):
            I_tag_val = example[i][constants.LABEL_TO_ID_E2E[f"I_{polarity}"]]
            if I_tag_val == 0 and len(tags_for_polarity) > 0:
                tags_for_polarity = [add_end_to_tag(
                    tag, i) for tag in tags_for_polarity]
                total_tags += tags_for_polarity
                tags_for_polarity = []

            B_tag_val = example[i][constants.LABEL_TO_ID_E2E[f"B_{polarity}"]]
            if B_tag_val == 1:
                tags_for_polarity.append({"start": i, "polarity": polarity})

    return total_tags


def calculate_popular_metrics(predictions, true_labels):
    tp, tn, fp, fn = compute_metrics_for_subset(predictions, true_labels)

    metrics = {}

    metrics["tp"] = tp
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["accuracy"] = divide((tp+tn), (tp+tn+fp+fn))
    metrics["precision"] = divide(tp, tp + fp)
    metrics["recall"] = divide(tp, tp + fn)
    metrics["f1"] = divide(2 * metrics["precision"] *
                           metrics["recall"], metrics["precision"] + metrics["recall"])
    return metrics


def calculate_f1_micro(metrics):
    tp_total = sum([metrics[f"tp_{pol}"] for pol in constants.POLARITIES])
    fp_total = sum([metrics[f"fp_{pol}"] for pol in constants.POLARITIES])
    fn_total = sum([metrics[f"fn_{pol}"] for pol in constants.POLARITIES])
    tn_total = sum([metrics[f"tn_{pol}"] for pol in constants.POLARITIES])

    precision_total = divide(tp_total, (tp_total + fp_total))
    recall_total = divide(tp_total, (tp_total + fn_total))

    return divide(2 * (precision_total * recall_total), (precision_total + recall_total))


def compute_metrics_E2E(results, cross_idx):

    def compute_metrics(p):
        predictions, true_labels = p
        predictions = predictions[:, :, :-1]
        true_labels = true_labels[:, :, :-1]

        save_pred_and_labels(predictions, true_labels, results, cross_idx)
        predictions = np.where(predictions > 0, np.ones(
            predictions.shape), np.zeros(predictions.shape))

        # Convert prediction to phrases
        predictions = [get_predicted_phrases(
            example) for example in predictions]
        true_labels = [get_predicted_phrases(
            example) for example in true_labels]

        metrics = {}
        # Calculate metrics for each polarity
        for polarity in constants.POLARITIES:
            pol_metrics = calculate_popular_metrics([[tag for tag in pred if tag["polarity"] == polarity] for pred in predictions], [
                                                    [tag for tag in label if tag["polarity"] == polarity] for label in true_labels])
            metrics.update(
                {f"{metric}_{polarity}": pol_metrics[metric] for metric in pol_metrics.keys()})

        # Total metrics
        metrics["f1_micro"] = calculate_f1_micro(metrics)
        metrics["f1_macro"] = sum(metrics[key] for key in [
                                  f"f1_{pol}" for pol in constants.POLARITIES]) / len(constants.POLARITIES)


        metrics.update(calculate_popular_metrics(predictions, true_labels))

        return metrics

    return compute_metrics
