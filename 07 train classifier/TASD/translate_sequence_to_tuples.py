import numpy as np
import constants
import re

# This function checks if/which aspect category was mentioned at the beginning of the prediction


def extract_aspect_category_from_string(sub_sentence):
    for ac in constants.AC_GERMAN.keys():
        if sub_sentence.startswith(ac):
            return ac
    return None


def extract_polarity_from_string(sub_sentence):
    for polarity in constants.POLARITY_GERMAN.keys():
        if sub_sentence.startswith("ist "+polarity):
            return constants.POLARITY_GERMAN[polarity]
    return None


def extract_aspect_term_from_string(sub_sentence):
    pattern = fr"weil (.*?)({'|'.join(constants.POLARITY_GERMAN.keys())}) ist"
    match = re.search(pattern, sub_sentence)
    if match:
        aspect_term = match.group(1).strip()
        return aspect_term
    else:
        return None


def get_single_tuple(sub_sentence):
    aspect_category = aspect_polarity = aspect_term = None
    is_explicit = True
    is_valid_example = False

    aspect_category = extract_aspect_category_from_string(sub_sentence)
    if aspect_category != None:
        aspect_polarity = extract_polarity_from_string(
            sub_sentence[len(aspect_category):].strip())
        if aspect_polarity != None:
            aspect_term = extract_aspect_term_from_string(sub_sentence)
            if aspect_term == "es":
                is_explicit = False

    if aspect_category != None and aspect_polarity != None and aspect_term != None:
        is_valid_example = True

    try:
        aspect_category = constants.AC_GERMAN[aspect_category]
    except:
        pass
    return {"aspect_term": aspect_term, "aspect_category": aspect_category, "aspect_polarity": aspect_polarity, "is_explicit": is_explicit, "is_valid_example": is_valid_example}


def get_tuples_from_prediction(example):
    split_example = [ex.strip() for ex in example.split("[SSEP]")]
    split_example = [item for item in split_example if not item.isspace()]
    split_example = [s for s in split_example if s.strip() != ""]

    tuples = []
    for sub_sentence in split_example:
        tuples.append(get_single_tuple(sub_sentence))

    return tuples
