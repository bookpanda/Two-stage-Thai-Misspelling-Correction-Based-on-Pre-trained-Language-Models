# WanchanBERTa Thai Grammarly
Correcting misspellings in large quantities can be arduous and tiresome, especially when the language is not used worldwide. Ignoring misspellings can be troublesome when training an NLP model for it could interpret two words intended to be the same as two completely different words just because of the spellings. This is why it is crucial to correct misspellings within any dataset before using it. With all of that said, I've spent some 8 weeks to try to develop an  NLP model to correct misspellings in Thai text.

You can read more in this <a href="https://medium.com/@marginpankam/wanchanberta-thai-grammarly-5010671797c7" target="_blank">Medium blog</a>.

# Directories
## /tpth
This is the code for the model trained with VISTEC_TPTH_2021 dataset

## /ugwc
This is the code for the model trained with Thai UGWC dataset (the public dataset was released as a json file so I named the code files with "json" suffixes at the time).

## File names
- data: making dataset for all models
- tagging: training and testing the model that does NER on input tokenized text
- masking: training and testing the model that receives the output of the tagging model and does MLM to predict misspelling corrections
- model: testing of tagging and masking on end-to-end tasks
- hunspell: competitor model using Hunspell
- pythainlp: competitor model using PyThaiNLP's misspelling collection function
- mt5: masking model but trained from mT5 base model
- CED: models with Character Edit Distance implementation in the post-proccessing after the masking model
