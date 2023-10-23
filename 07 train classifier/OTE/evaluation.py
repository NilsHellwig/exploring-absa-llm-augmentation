from helper import compute_popular_metrics
import numpy as np
import constants


def one_hot_to_label(one_hot):
    return next(constants.ID_TO_LABEL_OTE[idx] for idx in range(len(one_hot)) if one_hot[idx] == 1)


def find_bio_phrases(bio_list):
    phrases = []
    phrase_start = None

    for i in range(len(bio_list)):
        if bio_list[i] == 'B':
            if phrase_start is not None:
                phrase_end = i - 1
                phrases.append({"start": phrase_start, "end": phrase_end})
            phrase_start = i
        elif bio_list[i] == 'O':
            if phrase_start is not None:
                phrase_end = i - 1
                phrases.append({"start": phrase_start, "end": phrase_end})
                phrase_start = None

    if phrase_start is not None:
        phrases.append({"start": phrase_start, "end": len(bio_list) - 1})

    return phrases


def calculate_tp_tn_fp_fn_spans(pred, label):
    """
    Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) based on the provided
    lists of predicted and actual label ranges.

    Args:
        pred (list of dict): A list containing dictionaries representing predicted ranges with 'start' and 'end' values.
        label (list of dict): A list containing dictionaries representing actual label ranges with 'start' and 'end' values.

    Returns:
        tuple: A tuple containing four values - TP (true positives), FP (false positives), and FN (false negatives).
    """
    # Convert ranges to string representations and create sets.
    pred_set = set(f"{range['start']}_{range['end']}" for range in pred)
    label_set = set(f"{range['start']}_{range['end']}" for range in label)

    # Calculate true positives by finding the intersection of the sets.
    tp_set = pred_set & label_set
    tp = len(tp_set)

    # Calculate false positives by subtracting the intersection from the predicted set.
    fp_set = pred_set - tp_set
    fp = len(fp_set)

    # Calculate false negatives by subtracting the intersection from the label set.
    fn_set = label_set - tp_set
    fn = len(fn_set)

    # Calculate true negatives by considering all possible pairs and subtracting TP, FP, and FN.
    total_possible_pairs = len(pred) * len(label)

    return tp, 0, fp, fn


def compute_metrics_for_subset(examples_predictions, examples_labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(examples_predictions)):
        example_pred = find_bio_phrases(
            [one_hot_to_label(p) for p in examples_predictions[i]])
        example_label = find_bio_phrases(
            [one_hot_to_label(p) for p in examples_labels[i]])
        example_metrics = calculate_tp_tn_fp_fn_spans(
            example_pred, example_label)
        tp += example_metrics[0]
        tn += example_metrics[1]
        fp += example_metrics[2]
        fn += example_metrics[3]
    return tp, tn, fp, fn



def calculate_f1_micro(metrics):
    tp_total = sum([metrics[f"tp_{ac}"] for ac in constants.ASPECT_CATEGORIES])
    fp_total = sum([metrics[f"fp_{ac}"] for ac in constants.ASPECT_CATEGORIES])
    fn_total = sum([metrics[f"fn_{ac}"] for ac in constants.ASPECT_CATEGORIES])
    precision_total = tp_total / (tp_total + fp_total)
    recall_total = tp_total / (tp_total + fn_total)

    return 2 * (precision_total * recall_total) / (precision_total + recall_total)

def compute_metrics_OTE(test_data):

    def compute_metrics(p):
        aspect_categories = test_data.aspect_categories
        predictions, true_labels = p

        metrics = {}

        for ac in constants.ASPECT_CATEGORIES:
            # Select all examples of a given aspect category
            ac_predictions = np.array([predictions[i] for i in range(
                len(predictions)) if aspect_categories[i] == ac])
            ac_predictions = (ac_predictions == ac_predictions.max(
                axis=2)[:, :, np.newaxis]).astype(int)

            ac_labels = np.array([true_labels[i] for i in range(
                len(true_labels)) if aspect_categories[i] == ac])

            tp, tn, fp, fn = compute_metrics_for_subset(
                ac_predictions, ac_labels)

            precision, recall, f1 = compute_popular_metrics(tp, tn, fp, fn)
            metrics[f"f1_{ac}"] = f1
            metrics[f"precision_{ac}"] = precision
            metrics[f"recall_{ac}"] = recall
            metrics[f"tp_{ac}"] = tp
            metrics[f"tn_{ac}"] = tn
            metrics[f"fp_{ac}"] = fp
            metrics[f"fn_{ac}"] = fn
            metrics[f"n_samples_{ac}"] = len(ac_predictions)

        # Calculate f1_micro via f1 scores of all classes
        metrics["f1_micro"] = calculate_f1_micro(metrics)

        # Calculate F1 macro score
        metrics["f1_macro"] = sum(metrics[key] for key in [f"f1_{ac}" for ac in constants.ASPECT_CATEGORIES]) / len(constants.ASPECT_CATEGORIES)
       
        # Calculate toal scores
        tp, tn, fp, fn = compute_metrics_for_subset((predictions == predictions.max(axis=2)[:, :, np.newaxis]).astype(int), true_labels)
        metrics["precision"], metrics["recall"], metrics["f1"] = compute_popular_metrics(tp, tn, fp, fn)

        return metrics

    return compute_metrics
