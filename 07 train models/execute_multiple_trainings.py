import subprocess

# Real Trainings

for num in [500, 1000, 2000]:
    for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
        for few_shot_setting in ["random", "fixed"]:
            command = f"python train_absa_model.py only_real {num} 0 {task} {few_shot_setting}"
            process = subprocess.Popen(command, shell=True)
            process.wait()

# Synth Trainings fixed

for n_synth in [475, 975, 1975]:
    for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
        for llm in ["Llama70B", "GPT-3"]:
            command = f"python train_absa_model.py {llm} {n_synth} 25 {task} fixed"
            process = subprocess.Popen(command, shell=True)
            process.wait()

# Synth Trainings random
for n_synth in [500, 1000, 1500]:
    for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
        for llm in ["Llama70B", "GPT-3"]:
            command = f"python train_absa_model.py {llm} {n_synth} 500 {task} random"
            process = subprocess.Popen(command, shell=True)
            process.wait()
