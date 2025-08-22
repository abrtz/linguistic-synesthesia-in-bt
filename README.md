# Probing Language Models: Preservation of Linguistic Synesthesia in Back-translation

This repository contains the code to:
- annotate linguistic synesthesia with GPT-4 (API key required).
- compute IAA based on multiple annotators' with aggregated Cohen's kappa, aggregated Observed Agreement and Fleiss' kappa on the annotations of linguistic synesthesia. 
- generate final corpus with instances in which all human annotators agree.
- translate the positive cases of linguistic synesthesia into German and Spanish and translate them back to English
- annotate linguistic synesthesia in back-translations with GPT-4 (API key required).
- calculate IAA after backtranslation with aggregated Cohen's kappa, aggregated Observed Agreement and Fleiss' kappa
- compute IAA between before and after back-translation.

## Project Structure

```
.
├── README.md
├── back-translation
│   ├── 01_synesthesia-yes-no_annotating_llm.ipynb
│   ├── 02_synesthesia-yes-no_compute-iaa.ipynb
│   ├── 03_synesthesia-translation-backtranslation-with-llm.ipynb
│   ├── 04_synesthesia-backtranslations_annotating_llm.ipynb
│   ├── 05_synesthesia-backtranslation_compute-iaa.ipynb
│   ├── 06_synesthesia-backtranslation_compute-iaa-before-after-bt.ipynb
│   ├── data
│   │   ├── ES_DE_backtranslations_GPT.csv
│   │   ├── backtranslations
│   │   │   ├── all_annotators_DE_deepL.csv
│   │   │   ├── all_annotators_DE_google.csv
│   │   │   ├── all_annotators_DE_gpt.csv
│   │   │   ├── all_annotators_ES_deepL.csv
│   │   │   ├── all_annotators_ES_google.csv
│   │   │   ├── all_annotators_ES_gpt.csv
│   │   │   ├── translations_annotations_DE_deepL.csv
│   │   │   ├── translations_annotations_DE_google.csv
│   │   │   ├── translations_annotations_DE_gpt.csv
│   │   │   ├── translations_annotations_ES_deepL.csv
│   │   │   ├── translations_annotations_ES_google.csv
│   │   │   └── translations_annotations_ES_gpt.csv
│   │   ├── final_synesthesia-yes-no_all-annotators.csv
│   │   ├── synesthesia-yes-no.csv
│   │   └── synesthesia-yes-no_all-annotators.csv
│   ├── final_corpus
│   │   └── final_corpus_synesthesia_yes_no.csv
│   ├── iaa
│   │   ├── group1.json
│   │   ├── iaa.py
│   │   └── pairwise.json
│   ├── iaa.txt
│   ├── llm_utils.py
│   └── utils_iaa.py
├── iaa_corpus
│   ├── README.md
│   ├── data
│   │   ├── 3-annotators
│   │   │   ├── emily-dickinson.csv
│   │   │   ├── shakespeare.csv
│   │   │   └── ts-eliot.csv
│   │   └── 6-annotators
│   │       ├── emily-dickinson.csv
│   │       ├── shakespeare.csv
│   │       └── ts-eliot.csv
│   ├── final_corpus
│   │   ├── final_corpus_synesthesia.csv
│   │   ├── final_dickinson_synesthesia.csv
│   │   ├── final_eliot_synesthesia.csv
│   │   └── final_shakespeare_synesthesia.csv
│   ├── iaa-3-ann.ipynb
│   ├── iaa-6-ann.ipynb
│   ├── requirements.txt
│   └── utils.py
└── requirements.txt
```

## Requirements

Local experiments were run in Python 3.11.8.
All required libraries and versions are provided in the requirements.txt file. Run the following command to install them.

`pip install -r requirements.txt`

## 1. IAA Calculation and Final Corpora Generation for Linguistic Synesthesia in Poems

`iaa_corpus/`

This repository contains the code to compute IAA based on multiple annotators' aggregated Cohen's kappa on the annotations of linguistic or literary synesthesia. 90 instances were retrieved from the work of three different authors, namely Shakespeare's Poems, Emily Dickinson's Poetry and T.S. Eliot's Poems, with 30 instances each. Two rounds of annotations were performed on the whole data, first with 3 annotators and later one with an extra 3 annotators. IAA is calculated for each author as well as a whole.

A README is provided in this subdirectory.

## 2. Confirming if Linguistic Synesthesia was Preserved in Back-translation

`back-translation/`

- data/: Contains the initial corpus of linguistic synesthesia, the corpus after all annotations are done, and a subdirectory `backtranslations` with the back-translations. <br>
- final_corpus/: Contains the final corpus of linguistic synesthesia cases that is then used to filter the instances to anntotate. <br>
- utils_iaa.py: File containing the functions to format the csv files and compute IAA. <br>
- llm_utils.py: File containing the functions to annotate and translate with GPT-4. API key required. <br>
- iaa.txt: File containing the print outputs for all computed IAA, from the initial corpus, the final and after backtranslations. <br>

### Usage

To run the annotations and translation with LLM, an API key is required.


The notebooks are numbered on the steps to take to replicate this experiment.


01: annotations with LLM <br>
02: compute IAA and create final synesthesia corpus <br>
03: translate into DE and ES and translate back to EN with LLM <br>
04: annotations of back-translations with LLM <br>
05: compute IAA on back-translations <br>
06: compute IAA comparing before and after scenarios
