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
import shutil
import torch
import json
import sys
import os

# Parameters

LLM_NAME = sys.argv[1]
N_REAL = int(sys.argv[2])
N_SYNTH = int(sys.argv[3])
TARGET = sys.argv[4]
if TARGET not in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
    raise ValueError("Error: Not a valid target")

LLM_SAMPLING = sys.argv[5]
if LLM_SAMPLING not in ["random", "fixed"]:
    raise ValueError("Error: Not a valid sampling strategy")

print(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING)

# Settings

# Change modelname if no synthetic data used for training
if N_SYNTH == 0:
    LLM_NAME = "only_real"

# Set seeds
torch.device(constants.DEVICE)
torch.manual_seed(constants.RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(constants.RANDOM_SEED)
set_seed(constants.RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(constants.RANDOM_SEED)

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="transformers.optimization")

# Disable Pycache
sys.dont_write_bytecode = True

# Code

# Create Folders for Results
folders = ['split_results', 'results_csv', 'results_json']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load Dataset
train_dataset, test_dataset, validation_dataset = load_dataset_folds(
    LLM_NAME, N_REAL, N_SYNTH, LLM_SAMPLING)

# Load Model
if TARGET == "aspect_category":
    results = train_ACD_model(
        LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset, validation_dataset)

if TARGET == "aspect_category_sentiment":
    results = train_ACSA_model(LLM_NAME, N_REAL, N_SYNTH, TARGET,
                               LLM_SAMPLING, train_dataset, test_dataset, validation_dataset)

if TARGET == "end_2_end_absa":
    results = train_E2E_model(LLM_NAME, N_REAL, N_SYNTH, TARGET,
                              LLM_SAMPLING, train_dataset, test_dataset, validation_dataset)

if TARGET == "target_aspect_sentiment_detection":
    results = train_TASD_model(LLM_NAME, N_REAL, N_SYNTH, TARGET,
                               LLM_SAMPLING, train_dataset, test_dataset, validation_dataset)

# Save Results
with open(f'results_json/results_{LLM_NAME}_real{N_REAL}_synth{N_SYNTH}_{TARGET}_{LLM_SAMPLING}.json', 'w') as json_file:
    json.dump(results, json_file)

df = pd.DataFrame([results])
df.to_csv(
    f'results_csv/results_{LLM_NAME}_real{N_REAL}_synth{N_SYNTH}_{TARGET}_{LLM_SAMPLING}.csv', index=False)

# Remove useless folders

# try:
#     shutil.rmtree("outputs")
# except:
#     pass
