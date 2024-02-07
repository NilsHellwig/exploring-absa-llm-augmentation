import pandas as pd
import datetime
import json
import os


def format_seconds_to_time_string(total_seconds):
    time_duration = datetime.timedelta(seconds=total_seconds)
    hours, remainder = divmod(time_duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_format = ""

    if hours > 0:
        time_format += f"{hours}h "

    if minutes > 0:
        time_format += f"{minutes}m "

    time_format += f"{seconds}s"

    return time_format


def divide(a: int, b: int):
    return a / b if b > 0 else 0


def compute_popular_metrics(tp, tn, fp, fn):
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def save_pred_and_labels(predictions, labels, results, cross_idx):
    file_name = f"evaluation_{results['LLM_NAME']}_{results['N_REAL']}_{results['N_SYNTH']}_{results['TARGET']}_{results['LLM_SAMPLING']}_{cross_idx}"
    file_path = os.path.join('split_results', f'{file_name}.json')

    with open(file_path, 'w') as json_file:
        json.dump({"predictions": predictions.tolist(),
                   "labels": labels.tolist()}, json_file)


def get_hyperparameters(task, sample_size):
    tsv_file_path = f"../10 hyperparameter tuning/optuna_20_gbert_{task}_{sample_size}.tsv"
    df = pd.read_csv(tsv_file_path, delimiter='\t')
    max_f1_micro_index = df['f1_micro'].idxmax()
    values_at_max_f1_micro = df.loc[max_f1_micro_index, [
        'learning_rate', 'num_train_epochs', 'per_device_train_batch_size', 'f1_micro', 'accuracy']]

    return {"n_epochs": int(values_at_max_f1_micro["num_train_epochs"]), "learning_rate": float(values_at_max_f1_micro["learning_rate"]), "batch_size": int(values_at_max_f1_micro["per_device_train_batch_size"])}
