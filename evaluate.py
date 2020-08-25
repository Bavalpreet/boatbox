import spacy
import pickle
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import json
# final=[]

# preparing data which is required for testing
# with open('/home/bavalpreet/Documents/spacy-ner-annotator-master/25augfile.json1') as f:
#     b=[line.split('\n', 1) for line in f]
#     for i in range(len(b)):
#     	try:
#     		temp=json.loads(b[i][0])
    		
#     		if (len(temp['labels'])==0) or temp['labels'][0][2]=='IRRELEVANT' :
#     			continue
#     		else:
#     			# print(temp['text'])
#     			dictionary_f_tup={'entities':temp['labels']}
#     			temp_tuple=(temp['text'],dictionary_f_tup)
#     			final.append(temp_tuple)




#     	except:
#     		continue

#function to evaluate the test data
def evaluate(model, examples):
  scorer = Scorer()
  for input_, annot in examples:
    # print(type(input_),type(annot),annot)
    doc_gold_text = model.make_doc(input_)
    gold = GoldParse(doc_gold_text, entities=annot['entities'])
    pred_value = model(input_)
    scorer.score(pred_value, gold)
  return scorer.scores

# #selecting 20% of data
# m = int((len(final)*20)/100)
# Test_Data = final[:m]

# Test_DATA = [("VHF-Connection-Antenna end Hey guys accidentally broke the coax connection at the antenna end of my Shakespeare Galaxy 5400 antenna while removing/twisting it from the mount. I've got the shrink tubing pulled back and am trying to figure out the fix. It looks like the shield was soldered to the outside of the brass/copper antenna and the primary wire was taped and run up few inches inside of the antenna. Does this sound right? I'll see if can take pic. Anyone know how to put this thing back together? know buying new one is always an option, but this is pretty good galaxy so want to try to solder it back together and give it go. I'm ok with the soldering part. I've got redundant radios so I'd like to fix this one", {"entities": [(82, 89, "BOAT_PART"), (399, 406, "BOAT_PART"), (124, 131, "BOAT_PART"), (324, 331, "BOAT_PART")]}), ("If you want reliable antenna, one that you can count on in an emergency, you have no chioce but to replace it. Sure, you might be able to solder it, but how are you going to test it?", {"entities": [(21, 28, "BOAT_PART")]}), ("maybe he has an swr meter. the center has to attach to an element inside the outer shield tube. good luck getting it apart to fix it. your better off with new antenna", {"entities": [(16, 25, "ACCESSORIES"), (159, 166, "BOAT_PART"), (83, 94, "ACCESSORIES")]})]

# Now reading the file back into a Python list object.
with open ('outfile', 'rb') as fp:
    Test_Data = pickle.load(fp)
# print(a[:2])

#loading a model
nlp = spacy.load('100itertill25aug')
# #print(type(nlp))

#calling a function passing model and test data as a parameter
test_result = evaluate(nlp, Test_Data)

# printing the results and length of train and test data
print(test_result)
print(len(Test_Data))