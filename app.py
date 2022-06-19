######################
# Import libraries
######################

from transformers import AutoModelForMaskedLM
from transformers import BertForTokenClassification
import streamlit as st
import pickle
import torch

# print(transformers.__version__)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

######################
# Page Title
######################

# image = Image.open("tee_face.png")

# st.image(image, use_column_width=True)

st.write(
    """
# WanchanBERTa Thai Grammarly

This app can help correct some mispelled Thai words

***
"""
)

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    return tokenizer

@st.cache(allow_output_mutation=True)
def load_tagging():
    model = BertForTokenClassification.from_pretrained('bookpanda/wangchanberta-base-att-spm-uncased-tagging')
    return model

@st.cache(allow_output_mutation=True)
def load_masking():
    model = AutoModelForMaskedLM.from_pretrained("bookpanda/wangchanberta-base-att-spm-uncased-masking")
    return model

tokenizer = load_tokenizer()

class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = load_tagging()

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
tagging_model = BertModel()

ids_to_labels = {0: 'f', 1: 'i'}

def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    c = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)
        else:
            try:
              label_ids.append(2)
            except:
                label_ids.append(-100)

        previous_word_idx = word_idx
    return label_ids

def evaluate_one_text(model, sentence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'][0].unsqueeze(0).to(device)

    input_id = text['input_ids'][0].unsqueeze(0).to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
    # print(f"input_ids: {input_id}")
    # print(f"attnetion_mask: {mask}")
    # print(f"label_ids: {label_ids}")

    logits = tagging_model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]
    # print(f"logits_clean: {logits_clean}")

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    return prediction_label

mlm_model = load_masking()

######################
# Input Text Box
######################

st.header("Enter Sample Text")
st.write("ประเทศเราผลิตและส่งออกยาสูบเยอะสุดในโลกจิงป่าวคับ")
st.write("ไปดิ..รอไร")
st.write("ขอบคุณค้า ☺️☺️☺️☺️")
st.write("5555เดะบอกอีกทีสอบเสร็จก่อน")
st.write("ก้อนมีรายละเอียดกาต้มน้ำเพิ่มเติมตามรูปนะคร้าบบ^^")
st.write("ก้อมันจริงนิ")
st.write("สวัสดีทุกคนน้าคร้าบ")

def user_input_features():
    sequence_input = "ประเทศเราผลิตและส่งออกยาสูบเยอะสุดในโลกจิงป่าวคับ"
    sequence = st.text_area("Sequence input", sequence_input, height=250)
    sequence = sequence.splitlines()
    sequence = " ".join(sequence)  # Concatenates list to string        
    data = {
        "texts": sequence,
    }
    return data
input_data = user_input_features()

text = input_data['texts'].lower()
ans = []
i_f = evaluate_one_text(tagging_model, text)
print(i_f)
a = tokenizer(text)
b = a['input_ids']
c = tokenizer.convert_ids_to_tokens(b)
print(c)
i_f_len = len(i_f)
for j in range(i_f_len):
  if(i_f[j] == 'i'):
    ph = a['input_ids'][j+1]
    a['input_ids'][j+1] = 25004
    print(tokenizer.decode(a['input_ids']))
    b = {'input_ids': torch.Tensor([a['input_ids']]).type(torch.int64).to(device), 'attention_mask': torch.Tensor([a['attention_mask']]).type(torch.int64).to(device)}
    token_logits = mlm_model(**b).logits
    mask_token_index = torch.where(b["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    st.write(f"{tokenizer.convert_ids_to_tokens(ph)} => {tokenizer.convert_ids_to_tokens(top_5_tokens[0])}")
    ans.append((j, top_5_tokens[0]))
    text = ''.join(tokenizer.convert_ids_to_tokens(a['input_ids']))
    # for token in top_5_tokens:
    #     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
    a['input_ids'][j+1] = ph

print(a)

for x,y in ans:
  a['input_ids'][x+1] = y

final_output = tokenizer.convert_ids_to_tokens(a['input_ids'])
final_output.remove('<s>')
final_output.remove('</s>')
if final_output and final_output[0] == '▁':
    final_output.pop(0)
final_output = ''.join(final_output)
final_output = final_output.replace("▁", " ")
final_output = final_output.replace("<pad>", "")
print(final_output)


st.write(
    """
***
"""
)

st.header("OUTPUT")
st.write(final_output)
