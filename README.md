# Exploring LLMs for Synthetic Data Generation in ABSA

<div align="center">

**Exploring Large Language Models for the Generation of Synthetic Training Samples for Aspect-Based Sentiment Analysis in Low Resource Settings**

Published in **Expert Systems with Applications (ESWA)** · Volume 261, 2025

[![Paper](https://img.shields.io/badge/Paper_Download-ESWA%202025-blue?style=for-the-badge&logo=googlescholar)](https://doi.org/10.1016/j.eswa.2024.125514)
[![Correspondence](https://img.shields.io/badge/Contact-Nils%20Hellwig-darkred?style=for-the-badge&logo=minutemailer)](mailto:nils-constantin.hellwig@ur.de)

---

**Nils Constantin Hellwig · Jakob Fehle · Christian Wolff**

Media Informatics Group, University of Regensburg, Germany

*✉ Correspondence to: [nils-constantin.hellwig@ur.de](mailto:nils-constantin.hellwig@ur.de)*  
`{nils-constantin.hellwig, jakob.fehle, christian.wolff}@ur.de`

---

</div>

> **Abstract:** Aspect-Based Sentiment Analysis (ABSA) is a fine-grained task in sentiment analysis, aiming to identify sentiment expressed towards specific aspects of an entity. This paper explores the use of Large Language Models (LLMs), specifically GPT-3.5-turbo and Llama-3-70B, for generating annotated data in Aspect-Based Sentiment Analysis (ABSA), aiming to address the scarcity of labelled datasets in the field. Two low-resource scenarios are considered, with 25 and 500 manually annotated examples available. In the 25-example scenario, adding synthetic examples generated through few-shot prompting resulted in F1 scores of 81.33 for Aspect Category Detection (ACD) and 71.71 for Aspect Category Sentiment Analysis (ACSA). For the 500-example scenario, synthetic data augmentation showed a notable gain only for the ACSA task, raising the F1 score from 84.54 to 86.70.

---

## 🚀 Overview

This repository contains the official implementation for generating synthetic training data for Aspect-Based Sentiment Analysis (ABSA) using Large Language Models (LLMs). It evaluates how LLM-generated samples can enhance model performance in extreme low-resource settings.

### Key Features
- **Synthetic Data Generation**: Framework for generating ABSA-annotated samples using GPT-3.5 and Llama models.
- **Low-Resource Scenarios**: Evaluation pipelines for 25 and 500 initial manually annotated examples.
- **Multi-Task Support**: Support for Aspect Category Detection (ACD) and Aspect Category Sentiment Analysis (ACSA).
- **Comprehensive Evaluation**: Scripts for training lightweight models and comparing "real-only" vs. "augmented" datasets.

## 📁 Repository Structure

- `01 corpus acquisition/`: Scripts for collecting and preparing the raw restaurant review datasets.
- `02 dataset split/`: Data processing and split creation for reproducible experiments.
- `04 llm synthesis/`: Core logic for LLM-based synthetic data generation and few-shot prompting.
- `07 train models/`: Training and evaluation modules for various ABSA subtasks.
- `08 report model performance/`: Notebooks for statistical analysis and result visualization.

## 🛠️ Setup & Usage

### Installation

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_lg
# For Llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

### Usage

1. **Synthesize Data**: Use the scripts in `04 llm synthesis/` to generate examples.
   ```bash
   python 02_create_synth.py [MODEL_ID] [SPLIT] [FEW_SHOT_SETTING]
   ```
2. **Train Models**: Train downstream models using the combined datasets in `07 train models/`.
   ```bash
   python train_absa_model.py [LLM_NAME] [N_REAL] [N_SYNTH] [TARGET] [SAMPLING] [WITH_TRANSLATION]
   ```

## 📜 Citation

```bibtex
@article{HELLWIG2025125514,
  title = {Exploring large language models for the generation of synthetic training samples for aspect-based sentiment analysis in low resource settings},
  journal = {Expert Systems with Applications},
  volume = {261},
  pages = {125514},
  year = {2025},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2024.125514},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417424023819},
  author = {Nils Constantin Hellwig and Jakob Fehle and Christian Wolff},
  keywords = {Natural language processing (NLP), Sentiment analysis (SA), Aspect-based sentiment analysis (ABSA), Large language models (LLMs), Synthetic data generation, Low-resource settings, Data augmentation},
  abstract = {Aspect-Based Sentiment Analysis (ABSA) is a fine-grained task in sentiment analysis, aiming to identify sentiment expressed towards specific aspects of an entity. This paper explores the use of Large Language Models (LLMs), specifically GPT-3.5-turbo and Llama-3-70B, for generating annotated data in Aspect-Based Sentiment Analysis (ABSA), aiming to address the scarcity of labelled datasets in the field. Two low-resource scenarios are considered, with 25 and 500 manually annotated examples available. In the 25-example scenario, adding synthetic examples generated through few-shot prompting resulted in F1 scores of 81.33 for Aspect Category Detection (ACD) and 71.71 for Aspect Category Sentiment Analysis (ACSA). For the 500-example scenario, synthetic data augmentation showed a notable gain only for the ACSA task, raising the F1 score from 84.54 to 86.70.}
}
```
