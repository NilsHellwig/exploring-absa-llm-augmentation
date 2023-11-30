from torch.utils.data import Dataset as TorchDataset
import constants


def format_aspect_category(ac_text):
    if ac_text == "SERVICE":
        return "Service"
    if ac_text == "FOOD":
        return "Essen"
    if ac_text == "PRICE":
        return "Preis"
    if ac_text == "AMBIENCE":
        return "Ambiente"
    if ac_text == "GENERAL-IMPRESSION":
        return "Allgemeiner Eindruck"

    raise ValueError("Invalid Aspect Category in Example")


def format_polarity(polarity):
    if polarity == "NEGATIVE":
        return "schlecht"
    if polarity == "POSITIVE":
        return "gut"
    if polarity == "NEUTRAL":
        return "ok"
    raise ValueError("Invalid Sentiment Polarity in Example")

# “Service general is bad because waiter is bad”. # great | ok | bad


def tag_to_text(tag):
    if tag["type"] == "label-explicit":
        aspect_term_text = tag["text"]
    else:
        aspect_term_text = "es"
    ac_text = format_aspect_category(tag["label"])
    polarity_text = format_polarity(tag["polarity"])
    return f"{ac_text} ist {polarity_text}, weil {aspect_term_text} {polarity_text} ist"


def encode_example(example, tokenizer):
    tags_examples = example["tags"]
    input_text = example["text"]
    output_text = ""
    if len(tags_examples) < 1:
        raise ValueError("No Sentiment Annotations")
    for tag in tags_examples:
        formatted_triple = tag_to_text(tag)
        output_text += f"{formatted_triple} [SSEP] "

    output_text = output_text[:-1]

    input_encodings = tokenizer(input_text,
                                #   padding=True,
                                truncation=True,
                                max_length=constants.MAX_TOKENS_TASD,
                                #   return_tensors="pt"
                                )

    output_encodings = tokenizer(output_text,
                                 #   padding=True,
                                 truncation=True,
                                 max_length=constants.MAX_TOKENS_TASD,
                                 #   return_tensors="pt"
                                 )

    input_encodings["labels"] = output_encodings["input_ids"]

    return input_encodings


class CustomDatasetTASD(TorchDataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        item["labels"] = self.labels[idx]

        return item

    def __len__(self):
        return len(self.labels)
