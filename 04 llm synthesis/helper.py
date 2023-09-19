def divide_equally(x, random):
    equally = x // 3
    remainder = x % 3
    values = [equally, equally, equally]

    for i in range(remainder):
        values[i] += 1

    random.shuffle(values)

    return values

def convert_ner_to_xml(ner_dict):
    text = ner_dict['text']
    tags = ner_dict['tags']
    tag_positions = [] 

    for tag in tags:
        start = tag['start']
        end = tag['end']
        label = tag['label']
        polarity = tag['polarity']
        tag_type = tag['type']

        if tag_type == 'label-explicit':
            tag_positions.append((start, f'<aspect-term aspect="{label}" polarity="{polarity}">'))
            tag_positions.append((end, '</aspect-term>'))

    tag_positions.sort(reverse=True, key=lambda x: x[0])

    xml_text = list(text)
    for position, tag in tag_positions:
        xml_text.insert(position, tag)

    return ''.join(xml_text)


def get_examples_for_aspects_in_label(unique_aspects, random, dataset):
    example_entries = []
    
    for aspect in unique_aspects:
        example_found = False
        while example_found == False:
            example = random.choice(dataset)
            if not(example["id"] in [ex["id"] for ex in example_entries]) and random.choice(example["tags"])["label"] == aspect:
                example_found = True
        example_entries.append(example)
        
    return [entry["id"] for entry in example_entries]    


def get_examples_as_text(examples):
    labels = [[(tag["label"], tag["polarity"]) for tag in example["tags"]] for example in examples]
    predictions = [convert_ner_to_xml(example) for example in examples]
    example_text = "\n"
    
    for i in range(len(examples)):
        example_text += "Label:" + str(labels[i])+"\nPrediction:"+predictions[i]+"\n"
    
    return example_text[:-2]