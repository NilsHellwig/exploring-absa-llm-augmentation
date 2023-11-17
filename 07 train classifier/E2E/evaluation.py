import numpy as np
import constants

def one_hot_to_label(one_hot):
    return next(constants.ID_TO_LABEL_E2E[idx] for idx in range(len(one_hot)) if one_hot[idx] == 1)



def compute_metrics_for_subset(examples_predictions, examples_labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    print(examples_labels[0])
    raise ValueError("This is a custom error message.")


def compute_metrics_E2E(test_data):

    def compute_metrics(p):
        predictions, true_labels = p
        np.save('test_data.npy', test_data)
        np.save('predictions.npy', predictions)
        np.save('true_labels.npy', true_labels)

        metrics = {}

        tp, tn, fp, fn = compute_metrics_for_subset(predictions, true_labels)

        return metrics

    return compute_metrics