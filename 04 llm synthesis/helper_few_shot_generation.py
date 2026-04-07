import xml.etree.ElementTree as ET
from collections import Counter
import os


def get_examples_for_classes(unique_aspects, dataset, random, n_examples_for_category):
    example_entries = []

    for aspect in unique_aspects:
        for i in range(0, n_examples_for_category):
            example_found = False
            while example_found == False:
                example = random.choice(dataset)
                if not (example["id"] in [ex["id"] for ex in example_entries]) and random.choice(example["tags"])["label"] == aspect:
                    example_found = True
            example_entries.append(example)

    return [entry["id"] for entry in example_entries]


def calculate_aspect_category_frequency(labels_example, CLASSES):
    category_count = {category: 0 for category in CLASSES}

    for label in labels_example:
        aspect, _ = label
        if aspect in category_count:
            category_count[aspect] += 1

    total_labels = len(labels_example)
    frequency_aspect_categories = {
        category: count / total_labels for category, count in category_count.items()}

    return frequency_aspect_categories


def round_dict_to_target_sum(label_ratio, size):
    rest_dict = {}
    int_dict = {}

    for key, value in label_ratio.items():
        integer_part = int(value)
        fractional_part = value - integer_part
        int_dict[key] = integer_part
        rest_dict[key] = fractional_part

    required_rest = size - sum(int_dict.values())
    sorted_rest_dict = sorted(
        rest_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_keys = [key for key, value in sorted_rest_dict[:required_rest]]

    for k in top_k_keys:
        int_dict[k] += 1
    return int_dict


def get_opinion_counts(filepath):

    tree = ET.parse(filepath)
    root = tree.getroot()

    opinion_counts = []

    for review in root.findall('.//sentence'):
        opinion_count = len(review.findall('.//Opinion'))
        opinion_counts.append(opinion_count)

    return opinion_counts


def get_label_ratio_fixed(size):
    total_opinion_counts = []

    for filename in os.listdir("semeval"):
        if filename.endswith(".xml"):
            file_path = os.path.join("semeval", filename)
            total_opinion_counts += get_opinion_counts(file_path)
        pass

    total_opinion_counts = [
        count for count in total_opinion_counts if count != 0]
    label_ratio = Counter(total_opinion_counts)
    n_opinions = sum([label_ratio[v] for v in label_ratio.keys()])

    for key in label_ratio.keys():
        label_ratio[key] = label_ratio[key] / n_opinions * size

    label_ratio = round_dict_to_target_sum(label_ratio, size)

    return label_ratio


def get_label_ratio_random(size, dataset):
    n_labels = [len(example["tags"]) for example in dataset]
    label_ratio = Counter(n_labels)
    label_ratio = round_dict_to_target_sum(label_ratio, size)
    for key in label_ratio.keys():
        label_ratio[key] = int(label_ratio[key] / len(dataset) * size)

    return label_ratio
