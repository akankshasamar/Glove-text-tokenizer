import pandas as pd
import joblib
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding


df=pd.read_csv('Raw/company_roads_categorised.csv')
docs=df['CompanyName']

labels=df['CategoryID']
labels = asarray(labels)
print labels.shape

t = Tokenizer()
t.fit_on_texts(docs)
joblib.dump(t,"Model/text_Tokenizer.pkl")
vocab_size = len(t.word_index)+1

encoded_docs = t.texts_to_sequences(docs)

max_length = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

embeddings_index=dict()

f = open('Datasets/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word]=coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = zeros((vocab_size,100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=30, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='sigmoid'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(padded_docs, labels, epochs=2, verbose=1)
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))	

#model.save('Model/company_classification.h5')
