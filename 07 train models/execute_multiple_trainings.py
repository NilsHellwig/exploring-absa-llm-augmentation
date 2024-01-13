import subprocess
import sys

# I used multiple terminal session at once to train all conditions.
terminal_session = sys.argv[1]

# Real Trainings

if terminal_session == 0:
    for num in [500, 1000, 2000]:
        for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
            for few_shot_setting in ["random"]:
                command = f"python train_absa_model.py only_real {num} 0 {task} {few_shot_setting}"
                process = subprocess.Popen(command, shell=True)
                process.wait()

    for n_synth in [1975]:
        for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
            for llm in ["Llama70B", "GPT-3"]:
                command = f"python train_absa_model.py {llm} 25 {n_synth} {task} fixed"
                process = subprocess.Popen(command, shell=True)
                process.wait()


# Synth Trainings random
if terminal_session == 1:
    for n_synth in [500, 1000, 1500]:
        for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
            for llm in ["Llama70B", "GPT-3"]:
                command = f"python train_absa_model.py {llm} 500 {n_synth} {task} random"
                process = subprocess.Popen(command, shell=True)
                process.wait()


# Synth Trainings fixed
if terminal_session == 2:
    for n_synth in [475, 975]:
        for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
            for llm in ["Llama70B", "GPT-3"]:
                command = f"python train_absa_model.py {llm} 25 {n_synth} {task} fixed"
                process = subprocess.Popen(command, shell=True)
                process.wait()
