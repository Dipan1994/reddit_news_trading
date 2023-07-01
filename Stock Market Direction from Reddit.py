#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


# In[2]:


DJI = pd.read_csv("C:\\Users\\dipan\\OneDrive\\Desktop\\upload_DJIA_table.csv")[['Date','Open','Adj Close']]
DJI = DJI.iloc[::-1]
DJI.reset_index(inplace=True,drop=True)
DJI['Return'] = DJI['Adj Close'].pct_change(1)
DJI['Label'] =  [1 if x>0 else 0 for x in DJI['Return']]
DJI


# In[3]:


df = pd.read_csv('C:\\Users\\dipan\\OneDrive\\Desktop\\Combined_News_DJIA.csv')
df['Label'] = DJI['Label']
df


# In[4]:


train = df[df['Date'] < '2015-01-01']
test = df[df['Date'] > '2014-12-31']


# In[5]:


# df.loc[1611:]


# In[6]:


#Bag of Words - BoW
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
BoW_vectorizer = CountVectorizer()
BoW_train = BoW_vectorizer.fit_transform(trainheadlines)


# In[7]:


#Logistic Regression
Log_Reg = LogisticRegression()
Log_Reg = Log_Reg.fit(BoW_train, train["Label"])


# In[8]:


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
BoW_test = BoW_vectorizer.transform(testheadlines)
predictions = Log_Reg.predict(BoW_test)


# In[9]:


pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])


# In[10]:


accuracy_score(test["Label"], predictions)  


# In[11]:


BoW_words = BoW_vectorizer.get_feature_names_out()
BoW_coeffs = Log_Reg.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : BoW_words, 
                        'Coefficient' : BoW_coeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf.head(10)


# In[12]:


#Model 1 pred
pred_BoW_LogReg = predictions


# In[13]:


#N-gram model
Ngram_vectorizer = CountVectorizer(ngram_range=(2,2))
Ngram_train = Ngram_vectorizer.fit_transform(trainheadlines)
Log_Reg_Ngram = LogisticRegression()
Log_Reg_Ngram = Log_Reg_Ngram.fit(Ngram_train, train["Label"])


# In[ ]:





# In[14]:


Ngram_test = Ngram_vectorizer.transform(testheadlines)
predictions = Log_Reg_Ngram.predict(Ngram_test)


# In[15]:


pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])


# In[16]:


accuracy_score(test["Label"], predictions)  


# In[17]:


Ngram_words = Ngram_vectorizer.get_feature_names_out()
Ngram_coeffs = Log_Reg_Ngram.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : Ngram_words, 
                        'Coefficient' : Ngram_coeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf.head(10)


# In[18]:


#Model 2
pred_Ngram_Log_Reg = predictions


# In[19]:


#Tokenization,Lemmatization, Stop Words and Punctuation
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().system('pip install gensim')
import gensim
from gensim.models import Word2Vec


# In[20]:


df.dropna(inplace=True)
# df.fillna('No text available')
text_columns = [col for col in df.columns if col.startswith('Top')]
df[text_columns] = df[text_columns].apply(lambda x: x.str.lower())

def remove_punctuation(text):
    punctuation = string.punctuation
    text = text.translate(str.maketrans("", "", punctuation))
    return text

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: remove_punctuation(x) if type(x) == str else x)

def remove_non_textual(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove special characters
    return text

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: remove_non_textual(x) if type(x) == str else x)

        
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words]
    return " ".join(words)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: remove_stopwords(x) if type(x) == str else x)
        
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df[text_columns] = df[text_columns].apply(lambda x: x.apply(lemmatize_words))

for col in text_columns:
    df[col + '_tokenized'] = df[col].apply(word_tokenize)


# In[21]:


def tfidf_features(dataframe):
    tfidf = TfidfVectorizer(min_df=2)  
    features = tfidf.fit_transform(dataframe)
    feature_names = tfidf.get_feature_names_out()
    return features, feature_names

df2 = df.copy()
for col in text_columns:
    if df[col].dtype == 'object':
        text = df[col].apply(str)  # convert pandas series to list of strings
        features, feature_names = tfidf_features(text)
        df_tfidf = pd.DataFrame(features.todense(), columns=feature_names)
        df2 = pd.concat([df2, df_tfidf], axis=1)


# In[22]:


# Word2Vec
# text = [text.split() for text in df['Top1_tokenized']]
model = Word2Vec(df['Top1_tokenized'], vector_size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv
vector = word_vectors['word']


# In[23]:


#Testing several classification algorithms
from sklearn.feature_extraction.text import CountVectorizer
text_data = df[['Top1_tokenized', 'Top2_tokenized', 'Top3_tokenized',
                'Top4_tokenized', 'Top5_tokenized', 'Top6_tokenized',
                'Top7_tokenized', 'Top8_tokenized', 'Top9_tokenized', 
                'Top10_tokenized','Top11_tokenized', 'Top12_tokenized',
                'Top13_tokenized', 'Top14_tokenized', 'Top15_tokenized',
                'Top16_tokenized','Top17_tokenized', 'Top18_tokenized',
                'Top19_tokenized', 'Top20_tokenized','Top21_tokenized', 
                'Top22_tokenized', 'Top23_tokenized', 'Top24_tokenized', 'Top25_tokenized']]

text_data = text_data.apply(lambda x: " ".join(str(x) for x in x), axis=1)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)
y = df["Label"]


# In[113]:


X.shape


# In[24]:


# text_data.loc[1611:]
# df.iloc[1608:1612]


# In[25]:


from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X[:1608,:]
X_test =  X[1608:,:]
y_train = y.loc[:1610]
y_test = y.loc[1611:]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# In[26]:


train_data = X_train
train_labels = y_train
test_data = X_test
test_labels = y_test


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = [LogisticRegression(), SVC(), RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier(), DecisionTreeClassifier()]
model_names = ["Logistic Regression", "Support Vector Classifier", "Random Forest", "Gradient Boosting", "K-Nearest Neighbors", "Decision Tree"]

results=[]
pred_nltk_prepro_dict = {}
for model, model_name in zip(models, model_names):
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    pred_nltk_prepro_dict[model_name] = predictions
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    results.append((model_name, accuracy, precision, recall, f1))
    
for result in results:
    print("Model: {}\nAccuracy: {:.2f}%\nPrecision: {:.2f}%\nRecall: {:.2f}%\nF1 Score: {:.2f}%\n".format(result[0], result[1]*100, result[2]*100, result[3]*100, result[4]*100))


# In[28]:


# pred_nltk_prepro_dict


# In[29]:


##############Neural Networks################################################################
df = pd.read_csv('C:\\Users\\dipan\\OneDrive\\Desktop\\Combined_News_DJIA.csv')


# In[30]:


# df.isna().sum()
df.fillna('market remains same', inplace = True)


# In[31]:


train_df = df[df['Date'] < '2015-01-01']
test_df = df[df['Date']>'2014-12-31']


# In[32]:


# !pip install contractions
# !pip install spacy
# !python -m spacy download en_core_web_sm
import nltk
import spacy
import re
from bs4 import BeautifulSoup
import requests
import unicodedata
import contractions

nlp = spacy.load('en_core_web_sm')


# In[33]:


def remove_special_characters(text, remove_digits= True):
    #text = text.replace('$', 'currency')
    pattern = r'[^\w]+' if not remove_digits else r'[^a-zA-Z]'
    text = re.sub(pattern," ",text)
    text = re.sub(r'\s+',' ', text)
    return text

def remove_accented_characters(text):
    text =  unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode('utf-8','ignore')
    return text

def spacy_lemma(text):
    text = nlp(text)
    new_text = []
    words = [word.lemma_ for word in text]
    for small in words:
        if small == '-PRON-':
            pass
        else:
            new_text.append(small)

    return ' '.join(new_text)

def contractions_text(text):
    return contractions.fix(text)

def stop_words_removal(text, is_lower_case = False, stopwords = None):
    if stopwords == None:
        stopwords = nlp.Defaults.stop_words
    
    if not is_lower_case:
        text = text.lower()
    tokens = nltk.word_tokenize(text)
    new_token = []
    for i in tokens:
        if len(i)<=1:
            pass
        else:
            new_token.append(i)
    
    removed_text = [word for word in new_token if word not in stopwords]
    
    return ' '.join(removed_text)

def join_news(text):
    full_text = []
    for ind in range(len(text)):
        combine_text = []
        for col in range(2,len(text.columns[2:])+2):
            combine_text.append(text.iloc[ind,col])
        full_text.append(' '.join(combine_text))
    return full_text


# In[34]:


train_data = join_news(train_df)
test_data = join_news(test_df)


# In[35]:


import tqdm

def preprocessor_engine(text):
    corpus =[]
    for sent in tqdm.tqdm(text):
        sent = remove_accented_characters(sent)
        sent = contractions_text(sent)
        sent = remove_special_characters(sent)
        sent = spacy_lemma(sent)
        sent = stop_words_removal(sent)
        corpus.append(sent)
    return corpus


# In[36]:


train_data = preprocessor_engine(train_data)
train_data


# In[37]:


test_data = preprocessor_engine(test_data)


# In[38]:


import tensorflow as tf
tokenzer = tf.keras.preprocessing.text.Tokenizer(oov_token = '<UNK>')
tokenzer.fit_on_texts(train_data)
train_sequences = tokenzer.texts_to_sequences(train_data)
test_sequences = tokenzer.texts_to_sequences(test_data)
# train_sequences


# In[39]:


print("Vocabulary size ={}".format(len(tokenzer.word_index)))
print("Number of Documents={}".format(tokenzer.document_count))


# In[40]:


MAX_SEQUENCE_LENGTH = pd.Series(train_data).apply(lambda x : len(x.split())).max() + 1

train_pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')
test_pad_sequneces = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH, padding='post')


# In[41]:


y_train = train_df['Label']
y_test = test_df['Label']


# In[42]:


def metrics(y_true, y_pred):
    print('Accuracy Score:', round(accuracy_score(y_true, y_pred),2))
    print('\nClassification Score:\n', classification_report(y_true, y_pred))
    print('\nConfusion Matrix:\n', confusion_matrix(y_true, y_pred))


# In[43]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, LSTM, GRU, Bidirectional, Embedding, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping


# In[44]:


pred_deep_learning_dict = {}
def deep_model(layer_name,layer_nomen, epochs=50):
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    EMBEDDING_DIM = 300 #Dimension for dense embedding for each token
    VOCAB_SIZE = len(tokenzer.word_index)
    model = Sequential()
    model.add((Embedding(input_dim =VOCAB_SIZE+1,output_dim = EMBEDDING_DIM,input_length = MAX_SEQUENCE_LENGTH)))
    model.add((layer_name(256)))
    model.add((Dense(256,activation = 'relu')))
    model.add(Dense(1,activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',optimizer="adam",metrics =['accuracy'])
    model.summary()
    fit_the_model(model,layer_nomen, epochs=50)
    
def predictions(model,layer_nomen):
    train_pred = np.argmax(model.predict(train_pad_sequences),axis=1)
    test_pred = np.argmax(model.predict(test_pad_sequneces),axis=1)
    pred_deep_learning_dict[layer_nomen] = test_pred
    metrics(y_train, train_pred)
    metrics(y_test, test_pred)

def fit_the_model(model,layer_nomen, epochs=50):
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_pad_sequences, y_train, epochs=epochs, validation_data=(test_pad_sequneces, y_test),
              callbacks=[early_stop],verbose=0)
    predictions(model,layer_nomen)


# In[45]:


deep_model(LSTM,'LSTM')


# In[46]:


deep_model(GRU,'GRU')


# In[47]:


def stack_model(layer_name,layer_nomen, epochs=50):
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    EMBEDDING_DIM = 300
    VOCAB_SIZE = len(tokenzer.word_index)
    model = Sequential()
    model.add((Embedding(input_dim =VOCAB_SIZE+1,output_dim = EMBEDDING_DIM,input_length = MAX_SEQUENCE_LENGTH)))
    model.add((layer_name(256, return_sequences=True)))
    model.add((layer_name(128, return_sequences=False)))
    model.add((Dense(256,activation = 'relu')))
    model.add(Dense(2,activation = 'softmax'))

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics =['accuracy'])
    model.summary()
    fit_the_model(model,layer_nomen, epochs=50)


# In[48]:


stack_model(LSTM,'LSTM stack')


# In[49]:


stack_model(GRU,'GRU stack')


# In[50]:


def bidirect_model(layer_name,layer_nomen, epochs=50, dropout=False):
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    EMBEDDING_DIM = 300 #Dimension for dense embedding for each token
    VOCAB_SIZE = len(tokenzer.word_index)
    model = Sequential()
    model.add((Embedding(input_dim =VOCAB_SIZE+1,output_dim = EMBEDDING_DIM,input_length = MAX_SEQUENCE_LENGTH)))
    
    model.add(Bidirectional(layer_name(256, return_sequences=True)))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Bidirectional(layer_name(128, return_sequences=False)))
    if dropout:
        model.add(Dropout(0.2))
    model.add((Dense(256,activation = 'relu')))
    model.add(Dense(2,activation = 'softmax'))

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="adam",metrics =['accuracy'])
    model.summary()
    fit_the_model(model,layer_nomen, epochs=50)


# In[51]:


bidirect_model(LSTM,'LSTM bidirect')


# In[52]:


bidirect_model(GRU, 'GRU bidirect')


# In[54]:


pred_df = pd.DataFrame(pred_deep_learning_dict)
pred_df = pd.concat([pred_df,pd.DataFrame(pred_nltk_prepro_dict)],axis=1) 

pred_df['BoW Log Reg'] = pred_BoW_LogReg
pred_df['Ngram Log Reg'] = pred_Ngram_Log_Reg

pred_df.to_csv("C:\\Users\\dipan\\OneDrive\\Desktop\\Predicted Stock NLP.csv")


# In[ ]:


#Transformers
# !pip install transformers
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


train_data


# In[ ]:


# SEQ_LEN = pd.Series(train_data).apply(lambda x : len(x.split())).max() + 1
SEQ_LEN = 128


# In[ ]:


Xids = np.zeros((len(train_data),SEQ_LEN),dtype=np.int32)
Xmask = np.zeros((len(train_data),SEQ_LEN),dtype=np.int32)

for i, sequence in enumerate(train_data):
    tokens = tokenizer.encode_plus(sequence, max_length = SEQ_LEN,truncation=True,padding="max_length",
                                  add_special_tokens=True,return_token_type_ids=False,return_attention_mask=True,
                                  return_tensors='tf')
    Xids[i,:], Xmask[i,:] = tokens['input_ids'], tokens['attention_mask']

    
    
Xids_test = np.zeros((len(test_data),SEQ_LEN),dtype=np.int32)
Xmask_test = np.zeros((len(test_data),SEQ_LEN),dtype=np.int32)

for i, sequence in enumerate(test_data):
    tokens = tokenizer.encode_plus(sequence, max_length = SEQ_LEN,truncation=True,padding="max_length",
                                  add_special_tokens=True,return_token_type_ids=False,return_attention_mask=True,
                                  return_tensors='tf')
    Xids_test[i,:], Xmask_test[i,:] = tokens['input_ids'], tokens['attention_mask']


# In[ ]:


def map_fn(input_ids,masks,labels):
    return {'input_ids':input_ids, 'attention_mask':masks}, labels


# In[ ]:


labels = np.zeros((y_train.size,y_train.max()+1))
labels[np.arange(y_train.size),y_train] = 1
# train_tf = tf.data.Dataset.from_tensor_slices((Xids,Xmask,labels))
# train_tf = train_tf.shuffle(100000).batch(32)
# train_tf.map(map_fn)
# ds_len = len(list(train_tf))

test_labels = np.zeros((y_test.size,y_test.max()+1))
test_labels[np.arange(y_test.size),y_test] = 1
# test_tf = tf.data.Dataset.from_tensor_slices((Xids_test,Xmask_test,test_labels))
# test_tf.map(map_fn)

train_encodings = tokenizer.batch_encode_plus(train_data, truncation=True, max_length = SEQ_LEN,
                                        padding="max_length",return_tensors='tf')
# train_tf = tf.data.Dataset.from_tensor_slices((dict(train_encodings), labels))
# train_tf = train_tf.shuffle(100).batch(16)
train_iid = train_encodings['input_ids']
train_am = train_encodings['attention_mask']


test_encodings = tokenizer.batch_encode_plus(test_data, truncation=True, max_length = SEQ_LEN,
                                       padding="max_length",return_tensors='tf')
test_tf = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

# split = 0.9
# temp = train_tf
# train_tf = temp.take(round(ds_len*split))
# val_tf = temp.skip(round(ds_len*split))

# del temp



# In[ ]:


bert = TFAutoModel.from_pretrained('bert-base-uncased')
input_ids = tf.keras.layers.Input(shape=(SEQ_LEN), name = 'input_ids', dtype=tf.int32)
mask = tf.keras.layers.Input(shape=(SEQ_LEN), name = 'attention_mask', dtype=tf.int32)

embeddings = bert(input_ids, attention_mask = mask)[0]

X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128,activation = 'relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32,activation = 'relu')(X)
y = tf.keras.layers.Dense(2, activation = 'softmax', name = 'outputs')(X)

model  = tf.keras.Model(inputs = [input_ids,mask],outputs = y)
model.layers[2].trainable = False

optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
acc = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# model.compile(optimizer=optimizer, loss = loss, metrics=[acc])


# In[ ]:


with tf.GradientTape() as tape:
    outputs = model([train_iid, train_am])
    loss = loss_fn(labels, outputs)


# In[ ]:


history = model.fit(train_tf, validation_data = val_tf, epochs = 100)


# In[ ]:





# In[61]:


DJI_test = DJI[DJI['Date'] > '2014-12-31']
pred_df = pd.concat([DJI_test.reset_index(drop=True),pred_df],axis=1)


# In[93]:


Results = pd.DataFrame(np.cumprod(1+pred_df['Return']))
Results.columns = ['Holding']


# In[94]:


for i in pred_df.columns[5:]:
    Results[i] = np.cumprod(1+np.multiply([-1 if x==0 else 1 for x in pred_df[i]],pred_df['Return']))
    
Results['Date'] = pred_df['Date']


# In[98]:


Results.plot(x='Date', y=pred_df.columns[5:],kind="line", figsize=(14, 5))


# In[107]:


Results.plot(x='Date', y=['Holding','BoW Log Reg','Ngram Log Reg','Support Vector Classifier','Random Forest','Gradient Boosting',
                         'LSTM','LSTM bidirect'],kind="line", figsize=(14, 5), title = '$1 invested')


# In[ ]:





# In[ ]:




