from transformers import AutoTokenizer
from datasets import load_metric
import numpy as np
import constants
import nltk


def compute_metrics(eval_pred):
    # Evtl später übertragen
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_TASD)
    metric = load_metric("rouge")

    predictions, labels = eval_pred
    predictions = np.where(predictions != -100,
                           predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                       for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def calculate_metrics_for_example(label, prediction):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    remove_from_prediction = []
    for item in prediction:
        if item in label:
            tp += 1
            label.remove(item)
            remove_from_prediction.append(item)
        else:
            fp += 1
            remove_from_prediction.append(item)

    for item in remove_from_prediction:
        prediction.remove(item)

    for item in label:
        if not (item in prediction):
            fn += 1

    return tp, tn, fp, fn


def calculate_metrics_for_examples(labels, predictions):
    tp_total = 0
    tn_total = 0
    fp_total = 0
    fn_total = 0

    labels = [[{key: tuple[key] for key in tuple if key in ['aspect_term',
                                                            'aspect_category', 'aspect_polarity']} for tuple in label] for label in labels]
    predictions = [[{key: tuple[key] for key in tuple if key in [
        'aspect_term', 'aspect_category', 'aspect_polarity']} for tuple in pred] for pred in predictions]

    for i in range(len(predictions)):
        tp, tn, fp, fn = calculate_metrics_for_example(
            labels[i], predictions[i])
        tn_total += tn
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / \
        (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / \
        (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    accuracy = (tp_total + tn_total) / \
        (tp_total + tn_total + fp_total + fn_total)

    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return {"f1": f1, "recall": recall, "precision": precision, "accuracy": accuracy, "tp": tp_total, "tn": tn_total, "fp": fp_total, "fn": fn_total}
