import numpy as np
import constants 

def one_hot_to_label(one_hot):
    return next(constants.ID_TO_LABEL_ACD[idx] for idx in range(len(one_hot)) if one_hot[idx] == 1)   


def divide(a: int, b: int):
    return a / b if b > 0 else 0

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

def compute_metrics_OTE(test_data):
    
    def compute_metrics(p):
        aspect_categories = test_data.aspect_categories
        predictions, true_labels = p

        metrics = {}

        for ac in constants.ASPECT_CATEGORIES:
            examples_predictions = np.array([predictions[i] for i in range(len(predictions)) if aspect_categories[i] == ac])
            examples_labels = np.array([true_labels[i] for i in range(len(true_labels)) if aspect_categories[i] == ac])

            examples_predictions = (examples_predictions == examples_predictions.max(axis=2)[:,:,np.newaxis]).astype(int)

            examples_predictions = find_bio_phrases([one_hot_to_label(p) for example in examples_predictions for p in example])
            examples_labels = find_bio_phrases([one_hot_to_label(p) for example in examples_labels for p in example])
            
            tp, tn, fp, fn = calculate_tp_tn_fp_fn_spans(examples_predictions, examples_labels)
            precision = divide(tp, tp + fp)
            recall = divide(tp, tp + fn)
            f1 = divide(2 * precision * recall, precision + recall)
            metrics[f"f1_{ac}"] = f1
    
        metrics["f1_micro"] = sum(metrics[key] for key in [f"f1_{ac}" for ac in constants.ASPECT_CATEGORIES]) / (len(constants.ASPECT_CATEGORIES) - 1)

        return metrics

    return compute_metrics