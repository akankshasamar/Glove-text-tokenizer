from keras.models import load_model
import numpy
from keras.preprocessing.sequence import pad_sequences
import joblib
import pandas as pd

#df=pd.read_csv('cat_list.csv')
#print df
df=pd.read_csv('Company_Records.csv')
#print df.shape
#print type(df)
t=joblib.load('Model/text_Tokenizer.pkl')
#print type(t)
model=load_model('Model/company_classification.h5')
#print type(model)
#company=raw_input(df)
#ab=pd.DataFrame(df)
#print df.dtypes
ab=df['CompanyName'].apply(str)
company_sequence=t.texts_to_sequences(ab)
#print company_sequence
company_sequence_pad=pad_sequences(company_sequence, maxlen=30, padding='post')
# print company_sequence_pad
label= model.predict(company_sequence_pad,verbose=0)
#print label.shape
value=numpy.argmax(label, axis=1)
#print type(value)
#DataFrame.from_csv('Company_Records.csv')
values=list(value)
#print values
obj=pd.DataFrame({'col':values})
#print obj


#print my_list
result=pd.concat([df,obj],join='inner',axis=1)

print result[['CompanyName','col']]
new_list=result.to_csv('new_list.csv')
print new_list
