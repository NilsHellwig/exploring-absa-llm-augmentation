import constants
import random
import json

def load_dataset_folds(LLM_NAME, N_REAL, N_SYNTH, LLM_SAMPLING):
    # Load Real Dataset
    splits_real = []
    for i in range(constants.N_FOLDS):
        with open(f'real/split_{i}.json', 'r') as json_datei:
            real_split = json.load(json_datei)[:N_REAL]
            splits_real.append(real_split)
    # Load Synth Dataset
    splits_synth = []
    for i in range(constants.N_FOLDS):
        with open(f'synth/{LLM_NAME}/{LLM_SAMPLING}/split_{i}.json', 'r') as json_datei:
            synth_split = json.load(json_datei)[:N_SYNTH]
            splits_synth.append(synth_split)

    
    n_splits_required_real = constants.N_SPLITS_MAP.get(N_REAL, 0)
    n_splits_required_synth = constants.N_SPLITS_MAP.get(N_SYNTH, 0)
    n_splits_required_real, n_splits_required_synth

    # Five indexes, each for one cross valdiation run
    train_dataset = []
    test_dataset = []

    for i in range(constants.N_FOLDS):
        test_data = splits_real[i]
        train_data = []
    
        if N_REAL > 0:
            for split_idx in constants.SPLIT_LOOP[i+1: i+1+n_splits_required_real]:
                for example in splits_real[split_idx]:
                    train_data.append(example)
 
        if N_SYNTH > 0:
            for split_idx in constants.SPLIT_LOOP[i+1: i+1+n_splits_required_synth]:
                for example in splits_synth[split_idx]:
                    train_data.append(example)
                
        random.shuffle(train_data)
    
        train_dataset.append(train_data)
        test_dataset.append(test_data)

    return train_dataset, test_dataset