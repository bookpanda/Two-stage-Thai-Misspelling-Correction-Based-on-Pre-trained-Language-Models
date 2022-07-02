######################
# Import libraries
######################

import streamlit as st
import json
import random
import requests
import base64

######################
# Page Title
######################

st.write(
    """
# WanchanBERTa Thai Grammarly

This app can help correct some mispelled Thai words

***
"""
)

######################
# Input Text Box
######################

st.header("Enter Sample Text")
option = st.selectbox(
     'Example sentences',
     ('ประเทศเราผลิตและส่งออกยาสูบเยอะสุดในโลกจิงป่าวคับ',
      'ไปดิ..รอไร', 
      'ขอบคุณค้า ☺️☺️☺️☺️',
      '5555เดะบอกอีกทีสอบเสร็จก่อน',
      'ก้อนมีรายละเอียดกาต้มน้ำเพิ่มเติมตามรูปนะคร้าบบ^^',
      'ก้อมันจริงนิ',
      'สวัสดีทุกคนน้าคร้าบ'))

def user_input_features():
    sequence_input = option
    sequence = st.text_area("or just type it here (max characters: 60, excess characters will be removed)", sequence_input, height=250)
    sequence = sequence.splitlines()
    sequence = " ".join(sequence)  # Concatenates list to string        
    data = {
        "texts": sequence,
    }
    return data
input_data = user_input_features()

text = input_data['texts'].lower()
if len(text) > 60:
    text = text[:60]

url = "http://9a29-34-145-129-106.ngrok.io"

topost = {'text': text}    
post = requests.post(url, json=topost)
print(f"POST : {post}")
get_json = json.loads(post.text)

final_output = get_json

st.write(
    """
***
"""
)

st.header("OUTPUT")
st.write(final_output)
