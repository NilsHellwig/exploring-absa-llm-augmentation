import xml.etree.ElementTree as ET
from collections import Counter
import uuid
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from langdetect import detect

# Gültige Werte für aspect und polarity
VALID_ASPECT_VALUES = {"FOOD", "PRICE", "SERVICE", "GENERAL-IMPRESSION", "AMBIENCE"}
VALID_POLARITY_VALUES = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

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


def get_examples_as_text(examples):
    labels = [[(tag["label"], tag["polarity"]) for tag in example["tags"]] for example in examples]
    predictions = [convert_ner_to_xml(example) for example in examples]
    example_text = "\n"
    
    for i in range(len(examples)):
        example_text += "Label:" + str(labels[i])+'\nPrediction:'+predictions[i]+'\n'
    
    return example_text[:-2]

def remove_xml_tags(input_string):
    return re.sub(r'<[^>]+>', '', input_string)

def extract_aspect_polarity(xml_string):
    aspect_match = re.search(r'aspect="([^"]+)"', xml_string)
    polarity_match = re.search(r'polarity="([^"]+)"', xml_string)
    
    aspect = aspect_match.group(1) if aspect_match else None
    polarity = polarity_match.group(1) if polarity_match else None
    
    return aspect, polarity

def get_implicit_aspects(tags, predicted_text):
    
    # 1. match position
    pattern = r"<aspect-term(?!.*<aspect-term).*?<\/aspect-term>"
    matches = list(re.finditer(pattern, predicted_text))
    if len(matches) == 0:
        return tags, predicted_text  # Return tags and the updated text
    
    match = matches[0]
    
    # 2. position in text with tags
    tag_xml = match.group()
    tag_xml_start = match.start()
    tag_xml_end = match.end()
    
    # 3. identify aspect and polarity
    aspect, polarity = extract_aspect_polarity(tag_xml)
    
    # 4. position in text without tags
    tag_text = remove_xml_tags(tag_xml) 
    tag_start = len(remove_xml_tags(predicted_text[0:match.start()]))
    tag_end = tag_start + len(tag_text)
    
    # 5. remove tag from text
    predicted_text = predicted_text[0:tag_xml_start] + tag_text + predicted_text[tag_xml_end:]
    
    # 6. add tags to list
    tags.append({"text": tag_text, "start": tag_start, "end": tag_end, "tag_with_polarity": aspect+"-"+polarity, "tag_with_polarity_and_type": aspect+"-"+polarity+"-explicit", "type": "label-explicit", "label": aspect, "polarity":polarity})
    
    # Recursive call
    return get_implicit_aspects(tags, predicted_text)

    
def check_difference_between_tags_in_synth_text_and_label(label, tags_synth):
    """
    This function identifies the differences between aspect-polarity pairs in the label and the synthesised text.

    Args:
    label (list of tuples): The aspect-polarity pairs in the label.
    tags_synth (list of tuples): The aspect-polarity pairs in the synthesised text.

    Returns:
    tuple: A tuple containing two lists:
        - List of aspect-polarity pairs present in the label but not in the synthesised text.
        - List of aspect-polarity pairs present in the synthesised text but not in the label.
    """
    
    # Count the occurrences of aspect-polarity pairs in the label and synthesised text
    label_count = Counter(label)
    tags_synth_count = Counter(tags_synth)
    
    # Find aspect-polarity pairs in the label but not in the synthesised text
    not_in_tags_synth_count = [tup for tup, count in label_count.items() if count > tags_synth_count.get(tup, 0)]
    
    # Find aspect-polarity pairs in the synthesised text but not in the label
    not_in_label = [tup for tup, count in tags_synth_count.items() if count > label_count.get(tup, 0)]
    
    return not_in_tags_synth_count, not_in_label

def xml_to_json(xml_text, label, model_name, split_id):
    tags_synth, cleaned_text = get_implicit_aspects([], xml_text)
    tags_synth_in_label_format = [(tag["label"], tag["polarity"]) for tag in tags_synth]
    
    # Prüfen, ob alle identifizierten Tags im label
    not_in_tags_synth_count, not_in_label = check_difference_between_tags_in_synth_text_and_label(label, tags_synth_in_label_format)
    if len(not_in_label) > 0:
        return "not-in-label"
    
    
    # Add implicit aspects
    tags_synth = tags_synth + [{"text": None, 
                                "start": 0, 
                                "end": 0, 
                                "tag_with_polarity": tag[0]+"-"+tag[1], 
                                "tag_with_polarity_and_type":tag[0]+"-"+tag[1]+"-implicit",
                                "type": "label-implicit",
                                "label": tag[0],
                                "polarity": tag[1]
                               } for tag in not_in_tags_synth_count]
    
    
    return {"tags": tags_synth, "text": cleaned_text, "id": str(uuid.uuid4()), "model": model_name, "split": split_id}


def is_valid_xml(xml_string):
    try:
        ET.fromstring(xml_string)
        return True
    except ET.ParseError:
        return False

def elements_valid(elements):
    c = 0
    for element in elements:
        if element.tag == "aspect-term" and 'aspect' in element.attrib and 'polarity' in element.attrib:
            # Überprüfe, ob es andere Attribute gibt
            if set(element.attrib.keys()) - {'aspect', 'polarity'}:
                return False  # Es gibt andere Attribute

            # Überprüfe, ob die Werte für aspect und polarity gültig sind
            if element.attrib['aspect'] not in VALID_ASPECT_VALUES or element.attrib['polarity'] not in VALID_POLARITY_VALUES:
                return False  # Ungültige Werte für aspect oder polarity

            if elements_valid(element):
                c += 1
        else:
            return False  # Ein Element hat kein 'aspect' oder 'polarity' Attribut
    if c == len(elements):
        return True
    else:
        return False

def check_valid_aspect_xml(xml_string):
    try:
        elements = ET.fromstring(xml_string)
        return elements_valid(elements)
    except ET.ParseError:
        return False
    
def count_sentences_in_text(text):
    sentence_list = sent_tokenize(text)
    return len(sentence_list)


def german_language_detected(text):
    try:
        return detect(text) == "de"
    except:
        return False