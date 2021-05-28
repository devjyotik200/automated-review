import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot


data=pd.read_csv("scrapped_data_rmp.csv",engine='python')

for i in range(len(data)):
    if data['quality'][i]=="awful":
        data['quality'][i]=0
    else:
        data['quality'][i]=1

y=data['quality']
y=y.astype(int)
    

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

voc_size = 5000

for i in range(0, 2345):
    review = re.sub('[^a-zA-Z]', ' ', data['comment'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
onehot_rep = [one_hot(words,voc_size)for words in corpus]

embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=40)

X_final = np.array(embedded_docs)
y_final = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state = 0)

model = Sequential()
model.add(Embedding(voc_size,40,input_length=40))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


model.fit(X_final,y_final,epochs=20,batch_size=64)
model.save("model.h5")

y_pred = model.predict_classes(X_test)
y_pred2=model.predict_classes(X_final)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


data='Rand was a good guy, really... he was. But I felt he had an alter-ego when he left to go home and grade. His expectations and how he picked apart his students papers was simply not fair. Again, great guy but avoid this class with him!!!'
corpus1=[]
review1=re.sub('[^a-zA-Z]',' ',data)
review1= review1.lower()
review1 = review1.split()
review1 = [ps.stem(word) for word in review1 if not word in stopwords.words('english')]
review1 = ' '.join(review1)
corpus1.append(review1)
onehot_rep1 = [one_hot(words,5000)for words in corpus1]
embedded_docs1 = pad_sequences(onehot_rep1,padding='pre',maxlen=40)
test_final1 = np.array(embedded_docs1)
my_prediction=model.predict_classes(test_final1)
print(my_prediction)



