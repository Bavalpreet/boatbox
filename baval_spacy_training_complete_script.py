import json
import pickle
import random
import json

final=[]
#boatbox2
           
#READING DATA FROM THE JSON FILE
with open('/home/bavalpreet/Documents/spacy-ner-annotator-master/data_count/2septfile.json1') as f:
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


# #selecting 20% of data
count_of_test_data = int((len(final)*20)/100)
# print(m)

# randomly selecting 20 % data as a test data, it is returning a list named as m
m = random.choices(final, k=count_of_test_data)
# print(type(m),m)
print(len(m))
print(len(final))

# REMOVING TEST DATA FROM THE FINAL LIST
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

# NOW CHECKING WEATHER THE LENGTH OF FINAL IS REDUCED AND LENGTH OF TEST DATA
print("Length of train data",len(final), "Length of test data",len(m))

# VERIFYING THE INPUT FORMAT OF DATA
print(m[:2])

# saving the test data in the pickle file
with open('testdata2sept', 'wb') as fp:
    pickle.dump(m, fp)

finalbox2 = final[:]
print(len(final))
with open("/home/bavalpreet/Desktop/pickleinput_to_spacy/spacy_input_box2sept.data", "wb") as fp:
	pickle.dump(finalbox2, fp)
