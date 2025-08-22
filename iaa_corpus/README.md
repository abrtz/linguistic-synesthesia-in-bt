# IAA Calculation and Final Corpora Generation for Linguistic Synesthesia in Poems

This repository contains the code to compute IAA based on multiple annotators' aggregated Cohen's kappa on the annotations of linguistic or literary synesthesia. 90 instances were retrieved from the work of three different authors, namely Shakespeare's Poems, Emily Dickinson's Poetry and T.S. Eliot's Poems, with 30 instances each. Two rounds of annotations were performed on the whole data, first with 3 annotators and later one with an extra 3 annotators. IAA is calculated for each author as well as a whole. 

The final corpora is built from the files annotated by six annotators, totalling four final files: one per author and one containing all 90 labelled instances.

This repository contains the following files and directories:

- data/
- final_corpus/
- iaa-3-ann.ipynb
- iaa-6-ann.ipynb
- utils.py
- requirements.txt

In order to run both files, make sure libraries required are installed. They are listed in the requirements.txt file.
They can be installed with the following command:

`pip install -r requirements.txt`

- `data/`

This directory consists of two folders containing the files with the annotations from 3 annotators and 6 annotators respectively on each file with the literary synesthesia from each respective author.

- `final_corpus`

This directory contains the final corpora after computing IAA on six annotators and assigning a final gold label.

- `utils.py`

This python file contains the helper functions to format the transformed the data to instances per rows and annotations as columns, calculate the IAA agreement per annotator pairs, calculate the aggregated IAA with Cohen's kappa, get insight into the IAA value per file with all instances as well as regular and corner cases, plot the distribution of the labels and write the final corpus to csv.

- `iaa-3-ann.ipynb` & `iaa-6-ann.ipynb`

These notebooks call the helper functions in utlils after reading the files under `data/` repository. They compute the IAA agreement with 3 and 6 annotators respectively per file with all instances, corner and regular instances, and all files together.

`iaa-6-ann.ipynb` gets the final gold label from the annotations and writes the final corpora to the `final_corpus/` directory.



