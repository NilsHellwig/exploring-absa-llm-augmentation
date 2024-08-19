# Exploring Large Language Models for the Generation of Synthetic Training Samples for Aspect-Based Sentiment Analysis in Low Resource Settings

## Background

Sentiment analysis (SA), also named opinion mining, is a research area in natural language processing (NLP) which involves the computational classification of individuals' sentiments, opinions and emotions. This mostly involves categorizing sentiments into three polarities: positive, neutral and negative. Social media services like X (formerly known as Twitter) and Facebook have gained popularity as sources of data for SA due to the low barrier for posting messages, enabling a multitude of users around the world to express their opinions on various topics daily. Accordingly, content from social media is used for SA in many fields, for example in the political context, financial sector or healthcare.

Nevertheless, other data sources beside social media are also employed for SA. For example, customer reviews on websites, call center conversations or in-person interviews can serve as valuable sources for discerning user sentiment towards various topics, products and services. SA can be applied at both document- and sentence-level. However, if a document or sentence comprises a mixture of different sentiments, it might not be possible to assign only a positive, negative or neutral label. As an illustrative example, consider a sentence of a restaurant review wherein positive sentiment is expressed towards the food while, concurrently, negative sentiment is expressed when addressing the food's price. To address this issue, Aspect-Based Sentiment Analysis (ABSA) has been extensively studied as it goes beyond assessing general sentiment and instead delves into a more granular examination of sentiment by linking particular aspects with corresponding sentiment polarities.

## Objective

This study investigates the utilization of Large Language Models (LLMs), specifically GPT-3.5-turbo and Llama-3-70B, to generate annotated examples for Aspect-Based Sentiment Analysis (ABSA) in order to address the data scarcity of annotated datasets within the field of ABSA. Two low-resource scenarios were examined, considering pools of 25 and 500 manually annotated (real) examples that are given. For the generation of a new annotated example, 25 real examples were randomly selected and given as few-shot examples in the prompt. 

When considering a pool of 25 real examples for few-shot learning, the inclusion of synthetic training examples resulted in F1 scores of **81.33** and **71.71** for the Aspect Category Detection (ACD) and Aspect Category Sentiment Analysis (ACSA) tasks, respectively. In the case of a given pool of 500 real examples, data augmentation with synthetic examples didn't improve the performance, except for ACSA. For the ACSA task, the addition of examples generated with GPT-3.5-turbo significantly increased the F1 score from **84.54** to **86.70**.


## Installation



```bash
pip install -r requirements.txt
python -m spacy download de_core_news_lg
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
