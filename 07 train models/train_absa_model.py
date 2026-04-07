from load_dataset_folds import load_dataset_folds
from transformers import set_seed
from ACD import train_ACD_model
from E2E import train_E2E_model
from ACSA import train_ACSA_model
from TASD import train_TASD_model
import pandas as pd
import numpy as np
import constants
import warnings
import random
import shutil
import torch
import json
import sys
import os

# Training parameters provided via CLI
LLM_NAME = sys.argv[1]
N_REAL = int(sys.argv[2])
N_SYNTH = int(sys.argv[3])
TARGET = sys.argv[4]

# Validate training targets
VALID_TARGETS = ["aspect_category", "aspect_category_sentiment",
                 "end_2_end_absa", "target_aspect_sentiment_detection"]
if TARGET not in VALID_TARGETS:
    raise ValueError(
        f"Error: '{TARGET}' is not a valid target. Choose from: {VALID_TARGETS}")

LLM_SAMPLING = sys.argv[5]
if LLM_SAMPLING not in ["random", "fixed"]:
    raise ValueError(
        f"Error: '{LLM_SAMPLING}' is not a valid sampling strategy. Choose 'random' or 'fixed'.")

WITH_TRANSLATION = sys.argv[6].lower() == 'true'

print(
    f"Training Config: Model={LLM_NAME}, Real={N_REAL}, Synth={N_SYNTH}, Target={TARGET}, Sampling={LLM_SAMPLING}")

# Environment and Seed Configuration

# Update model name identifier if no synthetic data is used (baseline)
if N_SYNTH == 0:
    LLM_NAME = "only_real"

# Ensure deterministic behavior for GPU/CPU operations
torch.device(constants.DEVICE)
torch.manual_seed(constants.RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(constants.RANDOM_SEED)
set_seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(constants.RANDOM_SEED)

# Suppress verbose optimization warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="transformers.optimization")

# Prevent creation of __pycache__ folders for a cleaner workspace
sys.dont_write_bytecode = True

# Main Execution Flow

# Initialize directories for output artifacts
for folder in ['split_results', 'results_csv', 'results_json']:
    os.makedirs(folder, exist_ok=True)

# Load dataset based on provided parameters
train_dataset, test_dataset = load_dataset_folds(
    LLM_NAME, N_REAL, N_SYNTH, LLM_SAMPLING, random, with_translation=WITH_TRANSLATION)


# Trigger training for the specific ABSA task
if TARGET == "aspect_category":
    results = train_ACD_model(
        LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset)

elif TARGET == "aspect_category_sentiment":
    results = train_ACSA_model(
        LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset)

elif TARGET == "end_2_end_absa":
    results = train_E2E_model(
        LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset)

elif TARGET == "target_aspect_sentiment_detection":
    results = train_TASD_model(
        LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset)

# Persist evaluation metrics to disk
results_filename = f"results_{LLM_NAME}_real{N_REAL}_synth{N_SYNTH}_{TARGET}_{LLM_SAMPLING}"

with open(f'results_json/{results_filename}.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

df = pd.DataFrame([results])
df.to_csv(f'results_csv/{results_filename}.csv', index=False)
