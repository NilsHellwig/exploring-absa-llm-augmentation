# ------------------ General Settings ------------------
N_FOLDS = 5
SPLIT_LOOP = [0, 1, 2, 3, 4, 0, 1, 2, 3]
RANDOM_SEED = 43
ASPECT_CATEGORIES = ["GENERAL-IMPRESSION",
                     "FOOD", "SERVICE", "AMBIENCE", "PRICE"]
POLARITIES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

# ------------------ ⚠️⚠️⚠️ Delete Later ⚠️⚠️⚠️ ------------------
TEST_FOLDS = 1

# ------------------ ACD ------------------
MODEL_NAME_ACD = "deepset/gbert-large"
LEARNING_RATE_ACD = 5e-06
EPOCHS_ACD = 1
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
MODEL_NAME_OTE = "deepset/gbert-base"
MAX_TOKENS_OTE = 256
BATCH_SIZE_OTE = 16
EPOCHS_OTE = 1
LEARNING_RATE_OTE = 5e-06
EVALUATION_STRATEGY_OTE = "epoch"
OUTPUT_DIR_OTE = "outputs/output_OTE"
WEIGHT_DECAY_OTE = 0.01
