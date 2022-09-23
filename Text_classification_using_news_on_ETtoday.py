#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date
import time
import random


# In[3]:


user_agents = [
 "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
 "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
 "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
 "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
 "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
 "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
 "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
 "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
 "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
 "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
 "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
 "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
 "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
 "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    ]


# In[4]:


#crawl the news from ETtoday from 2021/5/1-2021/5/31
year = 2021
month = 5
categories = ['pet', 'health','travel']
codes = [8,21,11]
allinfo = {}
for code in range(len(codes)):
    title = []
    link = []
    category = []
    for day in range(1, 32):
        url = "https://www.ettoday.net/news/news-list-"+str(year)+"-"+str(month)+"-"+str(day)+"-"+ str(codes[code])+".htm"
        res = requests.get(url)
        bs = BeautifulSoup(res.text, "lxml")
        news = bs.select(".part_list_2")
        for element in news:
            for a in element.select("a"):
                title.append(a.text)
                link.append("https://www.ettoday.net"+a.get("href"))
            for em in element.select("em"):
                category.append(em.text)
    df = pd.DataFrame()
    df['title'] = title
    df['link'] = link
    df['category'] = category
    df = df.dropna()
    df = df.reset_index(drop=True)
    allinfo[categories[code]] = df
    #print(df.head(5))


# In[6]:


#merge the dataframes of different category
news = pd.DataFrame(allinfo[categories[0]])
for i in range(1, len(categories)):
    news = news.append(allinfo[categories[i]])
news = news.reset_index()


# In[7]:


## crawl the content of news
allcontent = []
link = list(news.link)
missing_index = []
for i in range(len(news)):
    headers = {
    "user-agent":random.choice(user_agents)
    }
    content = " "
    res = requests.get(link[i], headers = headers)
    soup = BeautifulSoup(res.content, "lxml")
    try:
        soup = soup.find("div", class_="story")
        for a in soup.find_all("p"):
            p = a.string
            if p != None:
                content = content + p + " "
        allcontent.append(content)
    except:
        missing_index.append(i)
        pass


# In[9]:


# if there are news that didn't get from crawling, drop the rows
if missing_index:
    news = news.drop(missing_index)
    news = news.reset_index()
news['content'] = allcontent


# ## CKIP tokenization

# In[29]:


from ckiptagger import WS
import re
# 載入模型
ws = WS("../data")
tokenized_list = []
for row in news['content']:
    #replace numbers, english, \n and \t with " "
    row = re.sub(r"[\n\t0-9a-zA-Z]*", " ", row)
    #replace non-Chinese and , by ""
    row = re.sub(r'[^\u4e00-\u9fff\,]', '', row)
    # split the row by ',' and return a list
    row = row.split(",")
    word_segmentation = ws(row,
                sentence_segmentation=True,
                segment_delimiter_set={'?', '？', '!', '！', '。', ',',   
                                   '，', ';', ':', '、'})
    string = ' '.join(str(item) for rows in word_segmentation for item in rows)
    tokenized_list.append(string)


# In[31]:


news['tokenization'] = tokenized_list


# In[36]:


from keras.preprocessing.text import Tokenizer
max_words = 5000
max_len = 500
tokenizer = Tokenizer(num_words=max_words)  ## 使用的最大詞語數為5000
tokenizer.fit_on_texts(news['tokenization'])

## 使用word_index屬性可以看到每個詞對應的編碼
## 使用word_counts屬性可以看到每個詞對應的頻數
# for index,item in enumerate(tokenizer.word_index.items()):
#     if index < 10:
#         print(item)
#     else:
#         break
# print("===================")  
# for index,item in enumerate(tokenizer.word_counts.items()):
#     if index < 10:
#         print(item)
#     else:
#         break


# ## One-hot encode labels and train test split

# In[118]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
labelencoder = LabelEncoder()
le_y = labelencoder.fit_transform(news['category']).reshape(-1,1)
ohe = OneHotEncoder()
y = ohe.fit_transform(le_y).toarray()
x = news['tokenization']
#Split the data to have 20% validation split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)


# In[119]:


from sklearn import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


# In[120]:


train_seq = tokenizer.texts_to_sequences(train_x)
test_seq = tokenizer.texts_to_sequences(test_x)
## 將每個序列調整相同的長度
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)
# print(train_seq_mat.shape, test_seq_mat.shape)


# In[121]:


inputs = Input(name='inputs',shape=[max_len])
## Embedding(詞匯表大小,vector長度,每個新聞的詞長)
layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128,activation="relu",name="FC1")(layer)
layer = Dropout(0.3)(layer)
layer = Dense(3,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
model.summary()
## multi-classification problem use categorical_crossentropy
model.compile(loss="categorical_crossentropy",optimizer=RMSprop(),metrics=["accuracy"])


# In[122]:


model_fit = model.fit(train_seq_mat,train_y,batch_size=128,epochs=10,
                      validation_split=0.2)


# In[134]:


import numpy as np
test_pre = model.predict(test_seq_mat)
## evaluate the performance of prediction
confusion_matrix = metrics.confusion_matrix(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1))
print("Confusion matrix: \n",confusion_matrix)


# In[133]:


scores = model.evaluate(test_seq_mat, test_y, verbose=0)
print("Out-of-sample loss: ", scores[0],"Out-of-sample accuracy:", scores[1])


# In[125]:


print(metrics.classification_report(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1)))


# In[130]:


# 繪製歷程圖
import matplotlib.pyplot as plt
def show_train_history(train_history):
    plt.figure(figsize=(6,5))
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.xticks([i for i in range(len(train_history.history['accuracy']))])
    plt.title('Train History')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('Accuracy.png')
    plt.show()

    plt.figure(figsize=(6,5))
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.xticks([i for i in range(len(train_history.history['loss']))])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('Loss.png')
    plt.show()
show_train_history(model_fit)


# In[132]:


try:
    model.save_weights("Ettoday_text classification_lstm.h5")
    print("success")
except:
    print("error")





