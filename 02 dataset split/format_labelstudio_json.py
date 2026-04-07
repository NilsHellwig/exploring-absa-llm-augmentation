import json

def format_json(data):
    formatted_data = []

    for raw_entry in data:
        tags = []
        two_or_more_sentences = False
        aspect_available_without_judgement = False
        
        for annotation in raw_entry["annotations"]:
            for result in annotation["result"]:
                if "labels" in result["value"]:
                    tag = {
                        "end": result["value"]["end"],
                        "start": result["value"]["start"],
                        "tag_with_polarity": result["value"]["labels"][0],
                        "tag_with_polarity_and_type": result["value"]["labels"][0] + result.get("from_name", "")[5:],
                        "text": result["value"]["text"],
                        "type": result.get("from_name", "")
                    }

                    if tag["type"] == "label-implicit":
                        tag["end"] = 0
                        tag["start"] = 0
                        tag["text"] = "NULL"
                        tag["tag_with_polarity"] = tag["tag_with_polarity"][:-10]
                    else:
                        tag["text"] = result["value"]["text"]

                    # Splitte den Tag
                    tag_parts = tag["tag_with_polarity"].rsplit("-", 1)
                    if len(tag_parts) == 2:
                        tag["label"], tag["polarity"] = tag_parts
                    else:
                        tag["label"] = tag_parts[0]
                        tag["polarity"] = ""

                    tags.append(tag)
                    
                if "choices" in result["value"]:
                    if result["from_name"] == "two_or_more_sentences":
                        two_or_more_sentences = True
                    if result["from_name"] == "aspect_available_without_judgement":
                        aspect_available_without_judgement = True
                        
        text = raw_entry["data"]["text"]         
        entry = {"tags": tags, "text": text, "two_or_more_sentences":two_or_more_sentences, "aspect_available_without_judgement": aspect_available_without_judgement}

        raw_entry["data"] = {key: value for key, value in raw_entry["data"].items() if not key.startswith("Unnamed")}

        for key, value in raw_entry["data"].items():
            entry[key] = value
            
        formatted_data.append(entry)

    return formatted_data