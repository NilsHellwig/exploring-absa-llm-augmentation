# Master Thesis: Data Augmentation with Large Language Models for Aspect-Based Sentiment Analysis in Low-Resource Scenarios 

In this repository, you will find all resources related to the master's thesis *Data Augmentation with Large Language Models for Aspect-Based Sentiment Analysis in Low-Resource Scenarios.*

## Background

Sentiment analysis (SA) is a research area in natural language processing (NLP) which involves the computational classification of individuals' sentiments, opinions and emotions. This mostly involves categorizing sentiments into three polarities: positive, neutral and negative. Social media services like X (now rebranded as Twitter) and Facebook have gained popularity as sources of data for sentiment analysis due to the low barrier for posting messages, enabling a multitude of users around the world to express their opinions on various topics daily. Accordingly, content from social media is used for sentiment analysis in many fields, for example in the political context, financial sector or healthcare.

Sentiment analysis can be applied at both document- and sentence-level. However, if a document or sentence comprises a mixture of different sentiments, it might not be possible to assign a positive, negative or neutral label. As an illustrative example, consider a sentence of a restaurant review wherein positive sentiment is expressed towards the food while, concurrently, negative sentiment is expressed when addressing the food's price. To address this issue, Aspect-Based Sentiment Analysis (ABSA) has been extensively studied as it goes beyond assessing general sentiment and instead delves into a more granular examination of sentiment by linking particular aspects or attributes with corresponding sentiment polarities.

## Objective

This work aims to apply LLMs for generating annotated examples for ABSA for the mitigation of data scarcity. A human-annotated dataset of 3,000 sentences from restaurant reviews posted on TripAdvisor in the German language (low-resource language in ABSA), serve as the foundation for this study. Using increasing amounts of examples synthesized with LLMs in addition to a given amount of 500 real examples for training smaller German language models based on BERT architecture, it is evaluated how the models' performance differs when using only real examples. 

For generating a synthetic example, 25 random samples are drawn from the 500 real examples, serving as few-shot examples. Additionally, a condition is examined, for which we assume a significantly smaller set of 25 real examples. In this case, we consistently employ these 25 examples as the few-shot examples for generating all synthetic examples.

## Installation


```
# only for schlaubox.de
pip install llvmlite --ignore-installed
pip install -r requirements.txt
python -m spacy download de_core_news_lg
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
