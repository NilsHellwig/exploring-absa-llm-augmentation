# ------------------ General Settings ------------------
N_FOLDS = 6
SPLIT_LOOP = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
RANDOM_SEED = 43
DEVICE = "cuda"
ASPECT_CATEGORIES = ["GENERAL-IMPRESSION",
                     "FOOD", "SERVICE", "AMBIENCE", "PRICE"]
POLARITIES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
ASPECT_CATEGORY_POLARITIES = [
    f"{ac}-{pol}" for ac in ASPECT_CATEGORIES for pol in POLARITIES]

N_SPLITS_MAP_REAL = {
    500: 1,
    1000: 2,
    1500: 3,
    2000: 4
}
EVALUATE_AFTER_EPOCH = False
N_EPOCHS_EARLY_STOPPING_PATIENCE = 25

# ------------------ ACD ------------------
MODEL_NAME_ACD = "deepset/gbert-large"
LEARNING_RATE_ACD = 2.01e-05
BATCH_SIZE_ACD = 32
MAX_TOKENS_ACD = 256
OUTPUT_DIR_ACD = "outputs/output_ACD"

# ------------------ ACSA ------------------
MODEL_NAME_ACSA = "deepset/gbert-large"
LEARNING_RATE_ACSA = 3.5e-05
BATCH_SIZE_ACSA = 8
MAX_TOKENS_ACSA = 256
OUTPUT_DIR_ACSA = "outputs/output_ACSA"

# ------------------ E2E ------------------
LABEL_TO_ID_E2E = {'B_POSITIVE': 0,
                   'B_NEUTRAL': 1,
                   'B_NEGATIVE': 2,
                   'I_POSITIVE': 3,
                   'I_NEUTRAL': 4,
                   'I_NEGATIVE': 5}

ID_TO_LABEL_E2E = {0: 'B_POSITIVE',
                   1: 'B_NEUTRAL',
                   2: 'B_NEGATIVE',
                   3: 'I_POSITIVE',
                   4: 'I_NEUTRAL',
                   5: 'I_NEGATIVE'}

MODEL_NAME_E2E = "deepset/gbert-large"
MAX_TOKENS_E2E = 256
BATCH_SIZE_E2E = 16
STEPS_E2E = 1500
LEARNING_RATE_E2E = 2e-5
OUTPUT_DIR_E2E = "outputs/output_E2E"
WEIGHT_DECAY_E2E = 0.01

# ------------------ TASD ------------------

MODEL_NAME_TASD = "t5-large"
MAX_TOKENS_TASD = 256
BATCH_SIZE_TASD = 16
LEARNING_RATE_TASD = 3e-4
EPOCHS_TASD = 20
OUTPUT_DIR_TASD = "outputs/output_TASD"
LOGGING_STRATEGY_TASD = "epoch"
METRIC_FOR_BEST_MODEL_TASD = "f1"
WEIGHT_DECAY_TASD = 0.01
AC_GERMAN = {'Service': "SERVICE",
             'Ambiente': "AMBIENCE",
             'Allgemeiner Eindruck': "GENERAL-IMPRESSION",
             'Preis': "PRICE",
             'Essen': "FOOD"}
POLARITY_GERMAN = {"gut": "POSITIVE",
                   "ok": "NEUTRAL",
                   "schlecht": "NEGATIVE"}
