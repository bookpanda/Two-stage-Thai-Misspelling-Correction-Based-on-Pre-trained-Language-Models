WanchanBERTa Text Fixer
Correcting misspellings in large quantities can be arduous and tiresome, especially when the language is not used worldwide. Ignoring misspellings can be troublesome when training an NLP model for it could interpret two words intended to be the same as two completely different words just because of the spellings. This is why it is crucial to correct misspellings within any dataset before using it. With all of that said, I've spent some 8 weeks to try to develop an  NLP model to correct misspellings in Thai text.
How did I develop this model?
Firstly, there are 2 models working together to correct misspellings: misspelling detector (tagging model) and misspelling corrector (masking model, also known as MLM model). The tagging model will label word tokens as a correctly-spelled token ('i' tag) or a misspelled token ('f' tag). Then, the MLM model will mask tokens with 'i' tags and will try to replace the masks with correctly-spelled version of the original misspellings.
Both models are an implementation of WanchanBERTa, a BERT based model for Thai language, developed by the researchers in Vidyasirimedhi Institute of Science and Technology, a research facility and university in Rayong province.
Choosing a dataset
Due to limited time, I could not create enough data by taking a normal Thai sentence dataset and labelling every single token with 'i' or 'f' tags. Hence, finding a Thai dataset with existing misspelled labels is vital for this project. Luckily, there is the VISTEC-TP-TH-2021 dataset, which contains around 50,000 sentences with tags, and one of the tags is a misspelled tag (for example, <msp value="ครับ">ค้าบ</msp>). We will be using this tag to teach the tagging model what words to tag as 'i' (misspelled).
The <msp></msp> tags
Misspellings in the VISTEC-TP-TH-2021 dataset have many variants:
- repeating charecters e.g. มากกกกกก (มาก), รักๆๆๆๆ (รัก ๆ)
- "ๆ" without spacing e.g. ขอบคุณๆ (ขอบคุณ ๆ)
- abbreviation without periods e.g. มิย (มิ.ย.)
- tone mark shift e.g. แป๊ป (แปป)
- typo e.g. อุหนุน (อุดหนุน)
- intentional typo e.g. นะ (น้า), ณ๊อง (น้อง)
- lazy typing e.g. อลัง (อลังการ), แบต (แบตเตอรี)
- etc.
Data preprocessing
For the sake of simplicity, I put all the data preprocessing of every model (competitor models included) into one python notebook. The dataset used is derived from the VISTEC-TP-TH-2021 dataset. After importing the dataset, I only chose samples that contain misspelled tags, for there is no reason to take samples that are correctly spelled. I also excluded samples with '^' or '$', because these 2 symbols will be used to demarcate the starts and ends of misspellings. All in all, I got 42,893 samples.
After that, I removed "|" and all the tags, and stored the correct variant of the misspellings:
สวัสดี|<msp value="ครับ">ค้าบ</msp>|พี่|<ne>จอม</ne>
สวัสดี^ครับ$พี่จอม
Next step is to convert these sentences into lists of tokenized texts and tags.
msp_list: list of tokenized texts with misspellings
crt_list: list of corrected tokenized texts 
if_list: list of tags of msp_list
During the process, I added tokens that aren't supposed to be separated as whole tokens to the tokenizer to make it recognize correctly spelled words as single whole tokens. This will prevent msp_list from not aligning correctly with crt_list for supplanting misspelled tokens with corrected ones will only take up 1 "token space". If the tokenized misspelling takes more than 1 "token space", when replacing it with corrected token, just pad the other "token space" with empty token "_".
Lastly, I created the dataset that would be used in models:
tagging model: token ids and attention masks of texts with misspellings, tags of those texts
masking model: token ids and attention masks of texts with misspellings, token ids of corrected texts
seq2seq model (competitor): list of texts with misspellings and corrected texts
Hunspell model (competitor): dataset of tagging model and masking model

The data preprocessing notebook can be found here
Tagging Model
This model is derived from Name-Entity-Recognition model. Instead of having many classes of entities, this model only has 2: misspelled tokens ('i' tag) and correct tokens ('f' tag). This model needs its inputs as lists of tokenized texts and labels ('i' and 'f' tags). There is no need to preprocess data, because we've already done that in the data preprocessing file. You can just load the data, train the tagging model, test it and save the model!
The tagging model notebook can be found here
Masking Model
This is a Mask-Language Model. After the input text has been passed through the tagging model, we will mask ONLY the tokens with 'i' tags and the model will try to replace the mask with a corrected version of the masked token. This model needs its inputs as lists of tokenized texts, attention masks and labels ('i' and 'f' tags)
The masking model notebook can be found here
Tagging + Masking model
After training the tagging model and the masking model, we can create a pipeline of the 2 models. Raw input text is passed through the tagging model first to be tagged with 'i' or 'f' tags. Then it is masked and predicted on tokens with 'i' tags by the masking model. The output is a corrected version of the input text.
The tagging + masking model notebook can be found here
Seq2Seq model (competitor)
This is a Sequence-to-Sequence NMT with attention mechanism model. I trained it on sequences of misspelled and corrected texts. However, I don't put much faith in this model, for it performed poorly in the task of correcting misspellings.
The Seq2Seq model notebook can be found here
Hunspell model (competitor)
Hunspell is a spell-checking program (that does not use Deep Learning). For this model, I tokenized the text and passed every token to Hunspell to check for misspellings. If Hunspell reckons a token as a misspelling, it will output a few suggestions of "correctly-spelled" words. I always took the first choice, which had the highest probability.
The Hunspell model notebook can be found here
Testing the models
There are 3 competing models: 
1. tagging + masking model
2. Hunspell model
3. seq2seq model
BLEU Score
Since BLEU scores compare the similarity between 2 texts, I compared 2 pairs with BLEU: original text to label text and predicted text to label text. If the predicted-label score is higher than the original-label one, it means that the model is making the original text closer to the label text (good results). On the other hand, if the original-label score is higher, the model is likely not improving the original text.
F1 Score
I calculate F1 scores on a token level, comparing among the predicted, the original and the label texts. Here are my definitions:
True Positive: token is corrected, and it matchs the label token
False Positive: token is corrected, but it does NOT match the label token
True Negative: token is NOT corrected, but it matchs the label token
False Negative: token is NOT corrected, and it does NOT match the label token
There will be an overwhelming number of True Negatives, because most tokens don't need to be corrected. Only a few tokens per sample need correcting.
Accuracy
accuracy = A / B
A = number of token changes that MATCH the corresponding label text's token
B = number of token changes on the original text

I tested each of the models on the same 1,000 samples sentences.
Tagging + Masking model
This is the best result of all models (by a very great margin). Out of the 1,000 samples, 386 samples were changed and became closer to the label text, 304 samples were changed but got lower BLEU score, and 310 samples were NOT changed at all. From the accuracy and F1 scores, this model on average can correct around 25–28% of all misspellings to be exactly same as the label text. However, these metrics does not take into account cases of correcting misspellings into other synonyms. For instance, there was a case where the original token was "สบายๆ", and instead of changing it to "สบาย ๆ", the model changed it to "ง่าย ๆ". Though they are similar in meaning, the F1 and accuracy metrics did not take this into account.
Hunspell model (competitor)
With F1 score of 0.00014 and accuracy of 8.22*10^-5, it could be inferred that out of 1,000 sample sentences, there were only a few corrections that match the label text. This is because Hunspell's word suggestions are not as smart as today's AI-powered alternatives. For BLEU scores, every sample is closer to the label text before being passed through Hunspell.
As you can see, Hunspell tried to correct every word token, even the already correct ones. So, I pass the input through the tagging model before Hunspell to only choose tokens that need correcting. This is the result:
Now, the accuracy and F1 is simply 0. This is because the tagging model does not have 100% accuracy in detecting misspellings, so it may had missed a misspelling that Hunspell could have correctly correct.
Seq2Seq model (competitor)
This is probably the worst-performing model for misspelling correction task. Due to the specific input format and tokenizer, I could not calculate this model with accuracy, BLEU, or F1 metrics. Thus, I resorted to using the character error rate metric (CER). Like BLEU, I compared the original-label score to the predicted-label score. As you can see, the average CER score of the original text to the label text is 0.023, meaning they are very close in character level. However, the average CER score of the predicted text to the label text is a staggering 1.864, a very high error rate. Also, every sample has higher error rates after being passed through Seq2Seq model.
My Seq2Seq seemed to perform so poorly that it made input text unintelligible. This explains the very high CER scores of predicted text (which is not good).

How to use this model
