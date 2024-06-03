# Probing Language Models:
Preservation of Linguistic Synesthesia in Back-translation

This repository contains the code to:
- annotate linguistic synesthesia with ChatGPT API (key required).
- compute IAA based on multiple annotators' with aggregated Cohen's kappa, aggregated Observed Agreement and Fleiss' kappa on the annotations of linguistic synesthesia. 
- generate final corpus with instances in which all human annotators agree.
- translate the positive cases of linguistic synesthesia into German and Spanish and translate them back to English
- annotate linguistic synesthesia in back-translations with ChatGPT API (key required).
- calculate IAA after backtranslation with aggregated Cohen's kappa, aggregated Observed Agreement and Fleiss' kappa
- compute IAA between before and after back-translation.


## File Structure

data/: Contains the initial corpus of linguistic synesthesia, the corpus after back all annotations are done, and a subdirectory `backtranslations` with the back-translations. <br>
final_corpus/: Contains the final corpus of linguistic synesthesia cases that is then used to filter the instances to anntotate. <br>
utils_iaa.py: File containing the functions to format the csv files and compute IAA. <br>
llm_utils.py: File containing the functions to annotate and translate with GPT-4. API key required. <br>
iaa.txt: File containing the print outputs for all computed IAA, from the initial corpus, the final and after backtranslations. <br>

## Usage

To run the annotations and translation with LLM, an API key is required.


The notebooks are numbered on the steps to take to replicate this experiment.


01: annotations with LLM <br>
02: compute IAA and create final synesthesia corpus <br>
03: translate into DE and ES and translate back to EN with LLM <br>
04: annotations of back-translations with LLM <br>
05: compute IAA on back-translations <br>
06: compute IAA comparing before and after scenarios
