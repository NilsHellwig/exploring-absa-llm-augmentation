def get_examples_for_classes(unique_aspects, dataset, random, n_examples_for_category):
    example_entries = []
    
    for aspect in unique_aspects:
        for i in range(0, n_examples_for_category):
            example_found = False
            while example_found == False:
                example = random.choice(dataset)
                if not(example["id"] in [ex["id"] for ex in example_entries]) and random.choice(example["tags"])["label"] == aspect:
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
    frequency_aspect_categories = {category: count / total_labels for category, count in category_count.items()}

    return frequency_aspect_categories

def generate_labels_from_ratio(label_ratio, n_examples=10):
    aspect_list = []
    for aspekt, score in label_ratio.items():
        repetitions = int(score * n_examples)
        aspect_list.extend([aspekt] * repetitions)

    while len(aspect_list) < n_examples:
        max_aspekt = max(label_ratio, key=label_ratio.get)
        aspect_list.append(max_aspekt)

    aspect_list = aspect_list[:n_examples]
    if len(aspect_list) != n_examples:
        raise Exception(f"Weniger als {n_examples} Beispiele aus der gegebenen ratio erzeugt.")
    return aspect_list

def get_random_examples_for_given_aspects(aspect_list, dataset, random):
    example_entries = []

    for aspect in aspect_list:
        example_found = False
        while example_found == False:
            example = random.choice(dataset)
            if not (example["id"] in [ex["id"] for ex in example_entries]) and random.choice(example["tags"])["label"] == aspect:
                example_found = True
        example_entries.append(example)

    return [entry["id"] for entry in example_entries]

def get_random_examples(n_examples, labels_example, dataset, random, CLASSES):
    ac_frequency = calculate_aspect_category_frequency(labels_example, CLASSES)
    balanced_labels_10 = generate_labels_from_ratio(ac_frequency, n_examples)
    examples_id_for_label = get_random_examples_for_given_aspects(balanced_labels_10, dataset, random)
    return examples_id_for_label