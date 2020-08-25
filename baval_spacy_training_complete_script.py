import json
import pickle
import random
import json

final=[]
#boatbox2
           

with open('/home/bavalpreet/Documents/spacy-ner-annotator-master/25augfile.json1') as f:
    b=[line.split('\n', 1) for line in f]
    for i in range(len(b)):
    	try:
    		temp=json.loads(b[i][0])
    		
    		if (len(temp['labels'])==0) or temp['labels'][0][2]=='IRRELEVANT' :
    			continue
    		else:
    			# print(temp['text'])
    			dictionary_f_tup={'entities':temp['labels']}
    			temp_tuple=(temp['text'],dictionary_f_tup)
    			final.append(temp_tuple)




    	except:
    		continue



count_of_test_data = int((len(final)*20)/100)
# print(m)

# randomly selecting 20 % data as a test data, it is returning a list named as m
m = random.choices(final, k=count_of_test_data)
# print(type(m),m)
print(len(m))
print(len(final))

for j in m:
    for i in range(0,len(final)):
        try:
       
            # print(final[i])
            if j == final[i]:
                final.pop(i)
            else:
                continue
        except:
            continue
# [i for i, j in zip(m, final) if i == j]

print(len(final))
print(m[:2])

# saving the test data in the pickle file
with open('testdata', 'wb') as fp:
    pickle.dump(m, fp)

finalbox2 = final[:]
print(len(final))
with open("/home/bavalpreet/Desktop/pickleinput_to_spacy/spacy_input_box3.data", "wb") as fp:
	pickle.dump(finalbox2, fp)
