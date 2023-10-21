# ------------------ General Settings ------------------
N_FOLDS = 5
SPLIT_LOOP = [0, 1, 2, 3, 4, 0, 1, 2, 3]
RANDOM_SEED = 43
ASPECT_CATEGORIES = ["GENERAL-IMPRESSION",
                     "FOOD", "SERVICE", "AMBIENCE", "PRICE"]
POLARITIES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

N_SPLITS_MAP_REAL = {
    500: 1,
    1000: 2,
    2000: 4
}


# ------------------ ⚠️⚠️⚠️ Delete Later ⚠️⚠️⚠️ ------------------
TEST_FOLDS = 1

# ------------------ ACD ------------------
MODEL_NAME_ACD = "deepset/gbert-base"  # ⚠️⚠️⚠️ Später ändern zu large
LEARNING_RATE_ACD = 5e-06
EPOCHS_ACD = 1  # 5
BATCH_SIZE_ACD = 16
MAX_TOKENS_ACD = 256
EVALUATION_STRATEGY_ACD = "epoch"
OUTPUT_DIR_ACD = "outputs/output_ACD"


# ------------------ OTE ------------------
LABEL_TO_ID_OTE = {
    'O': 0,
    'B': 1,
    'I': 2,
}
ID_TO_LABEL_OTE = {
    0: 'O',
    1: 'B',
    2: 'I',
}
MODEL_NAME_OTE = "deepset/gbert-base"  # ⚠️⚠️⚠️ Später ändern zu large
MAX_TOKENS_OTE = 256
BATCH_SIZE_OTE = 16
EPOCHS_OTE = 1
LEARNING_RATE_OTE = 5e-06
EVALUATION_STRATEGY_OTE = "epoch"
OUTPUT_DIR_OTE = "outputs/output_OTE"
WEIGHT_DECAY_OTE = 0.01

# ------------------ TASD ------------------

MODEL_NAME_TASD = "t5-base"  # ⚠️⚠️⚠️ Später ändern zu large
MAX_TOKENS_TASD = 256
BATCH_SIZE_TASD = 8
LEARNING_RATE_TASD = 4e-5
EPOCHS_TASD = 20
OUTPUT_DIR_TASD = "outputs/output_TASD"
EVALUATION_STRATEGY_TASD = "epoch"
LOGGING_STRATEGY_TASD = "epoch"
SAVE_STRATEGY_TASD = "epoch"
METRIC_FOR_BEST_MODEL_TASD = "rouge1"
WEIGHT_DECAY_TASD = 0.01
