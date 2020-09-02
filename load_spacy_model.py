# import spacy
# from spacy import displacy
# nlp = spacy.load('100itertill25aug')
# # print(type(nlp))
# text = input('enter the text : \n')
# doc = nlp(text)
# for ent in doc.ents:
# 	print(ent.text, ent.start_char, ent.end_char, ent.label_)
	
	
	
#testing #27aug
import pickle

import spacy
from spacy import displacy
nlp = spacy.load('100itertill2sept')

#READING A TEST DATA FROM A PICKLE FILE
with open ('testdata2sept', 'rb') as fp:
    Test_Data = pickle.load(fp)

result_check = []
for i in Test_Data:
	text = i[0]
	doc = nlp(text)
	for ent in doc.ents:
		m = ent.text, ent.start_char, ent.end_char, ent.label_
	result_check.append(m)
print(len(Test_Data),len(result_check))	
# print(Test_Data[3],result_check[3])

#lists for seperating actual data into two lists as text and an entities
Text = []
Actual_entities = []

#appending actual data into two lists seperatly 
for i in Test_Data:
	Text.append(i[0])
	Actual_entities.append(i[1]['entities'])
	

#creating a dataframe from a above THREE lists
d = {'Text':Text,'Actual_entities':Actual_entities,'Predicted_entities':result_check}

import pandas as pd
df = pd.DataFrame(d)
df.to_csv('entities_output_comparison2sept.csv')
