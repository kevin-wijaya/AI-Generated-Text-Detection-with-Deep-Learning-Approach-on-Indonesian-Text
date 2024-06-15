#########################################################################################
# LIBRARY ###############################################################################
#########################################################################################

import os, re
from transformers import logging as transformers_logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

transformers_logging.set_verbosity_error()

import numpy as np
from gensim.models import Doc2Vec
import tensorflow as tf, torch
from transformers import BertTokenizer, BertForSequenceClassification as BertClassifier
from flask import Flask, request, render_template

#########################################################################################
# INIT ALL MODEL ########################################################################
#########################################################################################

bert_path = 'indolem/indobert-base-uncased'
bilstm_path = 'predict/BI_LSTM.h5'
bigru_path = 'predict/BI_GRU.h5'
lstm_path = 'predict/LSTM.h5'
gru_path = 'predict/GRU.h5'
indobert_path = 'predict/IndoBERT.model'
d2v_path = 'doc2vec/Doc2Vec.d2v'

bert = BertClassifier.from_pretrained(os.path.join(os.getcwd(), 'models', bert_path), num_labels=2, output_attentions=False, output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained(os.path.join(os.getcwd(), 'models', bert_path))
finetune = torch.load(os.path.join(os.getcwd(), 'models', indobert_path), map_location=torch.device('cpu'))
bert.load_state_dict(finetune)

bilstm = tf.keras.models.load_model(os.path.join(os.getcwd(), 'models', bilstm_path))
bigru = tf.keras.models.load_model(os.path.join(os.getcwd(), 'models', bigru_path))
lstm = tf.keras.models.load_model(os.path.join(os.getcwd(), 'models', lstm_path))
gru = tf.keras.models.load_model(os.path.join(os.getcwd(), 'models', gru_path))

d2v = Doc2Vec.load(os.path.join(os.getcwd(), 'models', d2v_path))

#########################################################################################
# FUNCTION PROCESS ######################################################################
#########################################################################################

def preprocessing(text):
  text = text.replace('-', ' ')
  text = re.sub(r'[\r\xa0\t]', '', text)
  text = re.sub(r"http\S+|www\S+", '', text)
  text = re.sub(r'\b\w*\.com\w*\b', '', text)
  text = re.sub(r'\[.*?\]|\(.*?\}|\{.*?\}', '', text)
  text = re.sub(r'\b(\w+)/(\w+)\b', r'\1 atau \2', text)
  text = re.sub(r'@[A-Za-z0-9]+|#[A-Za-z0-9]+', '', text)
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\s+', ' ', text)
  text = text.replace('\n', ' ')
  text = text.strip(' ')
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  text = text.lower()
  return text

def encoder(text):
    global tokenizer
    return tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        padding=True,
        return_attention_mask = True,
        return_tensors = 'pt')

def vectorizer(text):
    global d2v
    text_vector = np.array([d2v.infer_vector(text.split(), epochs=20)])
    text_vector = np.reshape(text_vector, (text_vector.shape[0], text_vector.shape[1], 1))
    return text_vector

def model_transformers(text_encoded):
    global bert
    model = bert
    model.eval()
    inputs = {'input_ids': text_encoded['input_ids'],
              'attention_mask': text_encoded['attention_mask']}
    with torch.no_grad(): outputs = model(**inputs)
    return np.argmax(outputs.logits, axis=1).tolist()[0]

def model_neural(text_vector, model_used):
    global bilstm, bigru, lstm, gru
    if model_used == 'Bi-LSTM': return 1 if bilstm.predict(text_vector)[0][0] > 0.5 else 0
    elif model_used == 'Bi-GRU': return 1 if bigru.predict(text_vector)[0][0] > 0.5 else 0
    elif model_used == 'LSTM': return 1 if lstm.predict(text_vector)[0][0] > 0.5 else 0
    else: return 1 if gru.predict(text_vector)[0][0] > 0.5 else 0

#process
def predict(text, model_used):
    text = preprocessing(text)
    output = ''
    if model_used == 'IndoBERT':
        text_encoded = encoder(text)
        output = model_transformers(text_encoded)
    else:
        text_vector = vectorizer(text)
        output = model_neural(text_vector, model_used)
    print(f'{output}\n','='*89)    
    return str(output)      

#########################################################################################
# WEBSITE ###############################################################################
#########################################################################################

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form.get('text')
    model_used = request.form.get('model')
    print('='*89,f'\n{text}\n{model_used}')
    return predict(text, model_used)

if __name__ == '__main__':
    app.run(debug=True, port="7000")