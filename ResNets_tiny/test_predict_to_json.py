import pandas as pd
import json
import numpy as np

def test_predict_to_json(test_predict,test_id_json_file_path):
	''' 
		generate prediction json file of test set.
		test_predict---numpy narray
		test_id_json_file_path----path of IDs file of test set,contains 2 objects: "index" and "image_id"
	'''
	# read json file
	test_ids = pd.read_json(test_id_json_file_path,typ='frame',orient='table') #,typ='frame',orient='table'
	test_ids = test_ids.set_index('index')
	list_tmp=[]
	m = test_predict.shape[0]
	for i in range(m):
		item_predict_class = int(test_predict[i])                  # class(int)
		item_id = str(test_ids.loc['loc'+str(i)]['image_id'])      # image ID (str)
		list_tmp.append({"image_id":item_id,"disease_class":item_predict_class})
	# write to json
	with open('test.json', 'w') as f1:
		json.dump(list_tmp, f1)
	f1.close()


#################################       TEST    #######################################
# generate test_temp.json---------- just for test
l=[]
d = {}
for i in range(120):
	d[i]='id'+str(i)
	l.append({'index':'loc'+str(i),'image_id':d[i]})
with open('./test_temp.json','w') as f:
	json.dump(l,f)
f.close()
# generate test_predict---------- just for test
test_predict = np.ones(120)

test_predict_to_json(test_predict,'./test_temp.json')

