from helper_synthesis import get_examples_as_text, xml_to_json, is_valid_xml, check_valid_aspect_xml, count_sentences_in_text, has_empty_tag, has_aspect_term_of_invalid_pos_tags, remove_xml_tags_from_string
from IPython.display import clear_output
from dotenv import load_dotenv
from llama_cpp import Llama
import openai
import time
import json
import uuid
import os
import sys 


import time
print(time.strftime("%d.%m.%Y %H:%M:%S"))

############### Settings ###############

load_dotenv()

# Default values
SPLIT = 0
MODEL_ID = 0
FEW_SHOTS = "random"

# Check if command-line arguments are provided
if len(sys.argv) > 1:
    # Parse command-line arguments
    MODEL_ID = int(sys.argv[1])
    SPLIT = int(sys.argv[2])
    FEW_SHOTS = sys.argv[3]

DATASET_PATH = f'../07 train models/real/split_{SPLIT}.json'
LABELS_AND_EXAMPLES_PATH = f"few_shot_examples/few_shot_examples_{FEW_SHOTS}.json"

# LLM Settings
MAX_TOKENS = 512
CONTEXT_SIZE = 4096
TEMPERATURE = 0.5

# Set Seed
SEED = int(str(43) + str(SPLIT) + str(MODEL_ID))

N_RETRIES = 25

# Setup Classes/Polarities for Synthesis
CLASSES = ["GENERAL-IMPRESSION", "FOOD", "SERVICE", "AMBIENCE", "PRICE"]
POLARITIES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
COMBINATIONS = [(aspect, polarity)
                for polarity in POLARITIES for aspect in CLASSES]

STOP_CRITERIA = ["\n"]

MODELS = ["Llama13B", "Llama70B", "Falcon40B", "GPT-3"]
# 175B, 70B und 40B
MODEL_PATHS = {"Llama13B": "llm_models/llama-2-13b.Q4_0.gguf",
               "Llama70B": "llm_models/llama-2-70b.Q4_0.gguf", "Falcon40B": "llm_models/falcon-40b-Q4_K_S.gguf"}
MODEL_NAME = MODELS[MODEL_ID]

SYNTH_PATH = f"../07 train models/synth/{MODEL_NAME}/{FEW_SHOTS}/split_{SPLIT}.json"

############### Print Status ###############

print("Split:", SPLIT, "Model:",
      MODELS[MODEL_ID], "Few-Shot Setting:", FEW_SHOTS)


############### Code ###############

# Load existing examples
with open(SYNTH_PATH, 'r', encoding="utf-8") as file:
    existing_examples = json.load(file)

with open('../prompt_template.txt', 'r') as file:
    PROMPT_TEMPLATE = file.read()

with open(DATASET_PATH, 'r', encoding='utf-8') as json_file:
    dataset = json.load(json_file)

# Setup Model

if MODEL_NAME == "Llama70B":
    llm = Llama(model_path=MODEL_PATHS[MODEL_NAME], seed=SEED,
                n_gpu_layers=44, n_ctx=CONTEXT_SIZE, verbose=False, n_gqa=8)
    clear_output(wait=False)

    def llm_model(text):
        return llm(text, max_tokens=MAX_TOKENS, stop=STOP_CRITERIA, echo=True, top_p=1, temperature=TEMPERATURE)["choices"][0]["text"][len(text):]

if MODEL_NAME == "Llama13B" or MODEL_NAME == "Falcon40B":
    llm = Llama(model_path=MODEL_PATHS[MODEL_NAME], seed=SEED,
                n_gpu_layers=1, n_ctx=CONTEXT_SIZE, verbose=False)
    clear_output(wait=False)

    def llm_model(text):
        return llm(text, max_tokens=MAX_TOKENS, stop=STOP_CRITERIA, echo=True, top_p=1, temperature=TEMPERATURE)["choices"][0]["text"][len(text):]

if MODEL_NAME == "GPT-3":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def llm_model(text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": text}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=STOP_CRITERIA
        )
        return response.choices[0].message.content.strip()


# Load Labels and Examples

with open(LABELS_AND_EXAMPLES_PATH, 'r', encoding='utf-8') as json_file:
    labels_and_examples = json.load(json_file)[f"split_{SPLIT}"]

labels = labels_and_examples["labels_for_prediction"]
labels = [[(aspect, polarity) for aspect, polarity in sub_list]
          for sub_list in labels]

examples = labels_and_examples["few_shot_ids"]

print(len(labels))