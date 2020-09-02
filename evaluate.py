import spacy
import pickle
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import pandas as pd
import json
# final=[]


#function to evaluate the test data
def evaluate(model, examples):
  scorer = Scorer()
  for input_, annot in examples:
    # print(type(input_),type(annot),annot)
    doc_gold_text = model.make_doc(input_)
    gold = GoldParse(doc_gold_text, entities=annot['entities'])
    pred_value = model(input_)
    scorer.score(pred_value, gold)
    # print(type(pred_value), type(gold))
  return scorer.scores



# Now reading the file back into a Python list object.
with open ('testdata2sept', 'rb') as fp:
    Test_Data = pickle.load(fp)
# print(a[:2])

#loading a model
nlp = spacy.load('100itertill2sept')
# #print(type(nlp))

#calling a function passing model and test data as a parameter
test_result = evaluate(nlp, Test_Data)

# printing the results and length of train and test data
print(test_result)
print(len(Test_Data))

#27 aug
ents_p = []
ents_r = []
ents_f = []
ents_per_type = []


ents_p.append(test_result['ents_p'])
ents_r.append(test_result['ents_r'])
ents_f.append(test_result['ents_f'])
ents_per_type.append(test_result['ents_per_type'])


d = {'ents_p': ents_p, 'ents_r': ents_r, 'ents_f': ents_f, 'ents_per_type': ents_per_type}
df = pd.DataFrame(d)
df.to_csv('prf_entities_output2sept.csv')




# dummy = []

# #27 Aug
# def evaluate(model, examples):
#   scorer = Scorer()
#   for input_, annot in examples:
#     # print(type(input_),type(annot),annot)
#     doc_gold_text = model.make_doc(input_)
#     gold = GoldParse(doc_gold_text, entities=annot['entities'])
#     pred_value = model(input_)
#     scorer.score(pred_value, gold)
#     # print(type(pred_value), type(gold))
#   return scorer.scores



# # Now reading the file back into a Python list object.
# with open ('testdata', 'rb') as fp:
#     Test_Data = pickle.load(fp)
# # print(a[:2])

# #loading a model
# nlp = spacy.load('100itertill25aug')
# # #print(type(nlp))

# #calling a function passing model and test data as a parameter
# test_result = evaluate(nlp, Test_Data)

# # printing the results and length of train and test data
# # print(type(test_result), test_result)
# uas = []
# las = []
# las_per_type = []
# ents_p = []
# ents_r = []
# ents_f = []
# ents_per_type = []
# tags_acc = []
# token_acc = []
# textcat_score = []
# textcats_per_cat = []
# # print(type(test_result),test_result)

# ents_p.append(test_result['ents_p'])
# ents_r.append(test_result['ents_r'])
# ents_f.append(test_result['ents_f'])
# ents_per_type.append(test_result['ents_per_type'])


# d = {'ents_p': ents_p, 'ents_r': ents_r, 'ents_f': ents_f, 'ents_per_type': ents_per_type}
# df = pd.DataFrame(d)
# df.to_csv('prf_entities_output.csv')
