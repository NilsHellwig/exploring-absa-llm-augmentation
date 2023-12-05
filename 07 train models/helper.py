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
    file_name = f"evaluation_{ '_'.join(str(results[r]) for r in results)}_{cross_idx}"
    file_path = os.path.join('split_results', f'{file_name}.json')

    with open(file_path, 'w') as json_file:
        json.dump({"predictions": predictions.tolist(),
                   "labels": labels.tolist()}, json_file)
