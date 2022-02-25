import csv
import re
import matplotlib.pyplot as plt
import matplotlib
'''matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})'''
import pandas as pd
import seaborn as sns
import numpy as np


fieldnames = ['sentence','distance_from_beginning','distance_from_masked_token','target_prob','ordered_items','target_occupation',\
'count_attractors','relative_prob','relative_rank']
#dir = 'datasets/BackUpData/'
#dir = 'datasets/'
dir = './data/combined_data/VaryingKeyEntity/'

count_distractors = [0,1,2,3]#,4,5] #include zero distractor case


gpt2=False
model_list = ['BertBase','BertLarge','RobertaBase','RobertaLarge','GPT2Small','GPT2Medium','GPT2Large','GPT2XL']
#fig = plt.figure()
#fig.set_size_inches(w=8,h=4) 
plt.rcParams["legend.loc"] = 'upper right'
#print(plt.rcParams)

plt.rcParams.update({'font.size': 14.5})
plt.rcParams.update({'legend.fontsize':11.6})
#plt.rcParams["font."]
color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','x','d','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted']
#fig,ax =  plt.subplots(1,2)

index = 0
for model in model_list:
	accuracy = 0
	count = 0
	
	count_0_attractor = 0
	count_1_attractor = 0
	count_2_attractor = 0
	count_3_attractor = 0
	count_4_attractor = 0
	count_5_attractor = 0
	count_6_attractor = 0

	correct_0_attractor = 0
	correct_1_attractor = 0
	correct_2_attractor = 0
	correct_3_attractor = 0
	correct_4_attractor = 0
	correct_5_attractor = 0
	correct_6_attractor = 0
	correct = 0

	acc_count_attractor = []

	#for key entity occuring later. This corresponds to Fig:5
	file = dir+'/'+model+'/complete_data_For_MultipleEntityObjectDistractorAccuracy'+model+'.csv'
	f = open(file,'r')
	reader = csv.DictReader(f,delimiter='\t')
	#print('count ',count)
	for row in reader:
		item_list = re.sub(r'[^\w]', ' ', row['ordered_items'])
		item_list = item_list.split()



		
		if int(row['count_attractors']) == 1:
			if row['target_occupation'].lower() == item_list[0].lower():
				correct_1_attractor+=1
			count_1_attractor+=1
		if int(row['count_attractors']) == 2:
			if row['target_occupation'].lower() == item_list[0].lower():
				correct_2_attractor+=1
			count_2_attractor+=1

		if int(row['count_attractors']) == 3:
			if row['target_occupation'].lower() == item_list[0].lower():
				correct_3_attractor+=1
			count_3_attractor+=1
		if int(row['count_attractors']) == 4:
			if row['target_occupation'].lower() == item_list[0].lower():
				correct_4_attractor+=1
			count_4_attractor+=1
		if int(row['count_attractors']) == 5:
			if row['target_occupation'].lower() == item_list[0].lower():
				correct_5_attractor+=1
			count_5_attractor+=1

		if int(row['count_attractors']) == 0:
			if row['target_occupation'].lower() == item_list[0].lower():
				correct_0_attractor+=1
			count_0_attractor+=1



	accuracy_attractor = correct_0_attractor/float(count_0_attractor)
	acc_count_attractor.append(accuracy_attractor)

	accuracy_attractor = correct_1_attractor/float(count_1_attractor)
	acc_count_attractor.append(accuracy_attractor)

	accuracy_attractor = correct_2_attractor/float(count_2_attractor)
	acc_count_attractor.append(accuracy_attractor)

	accuracy_attractor = correct_3_attractor/float(count_3_attractor)
	acc_count_attractor.append(accuracy_attractor)

	'''accuracy_attractor = correct_4_attractor/float(count_4_attractor)
	acc_count_attractor.append(accuracy_attractor)

	accuracy_attractor = correct_5_attractor/float(count_5_attractor)
	acc_count_attractor.append(accuracy_attractor)'''



	##############################################################end of multi#################################################

	count_1_attractor_single_entity=0
	count_2_attractor_single_entity = 0
	count_0_attractor_single_entity=0
	count_3_attractor_single_entity = 0
	count_4_attractor_single_entity = 0
	count_5_attractor_single_entity = 0

	correct_1_attractor_single_entity = 0
	correct_2_attractor_single_entity = 0
	correct_0_attractor_single_entity = 0
	correct_3_attractor_single_entity = 0
	correct_4_attractor_single_entity = 0
	correct_5_attractor_single_entity = 0

	acc_count_attractor_single_entity = []


	plt.plot(count_distractors, acc_count_attractor, '-ok',color=color_list[index],\
		markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
	
	plt.xlabel('number of attractors', labelpad=1)
	plt.ylabel('accuracy', labelpad=1)
	#plt.title("MultipleEntityWithKeyEntityVarying")
	max_x = max(count_distractors)
	plt.xticks(np.arange(0, max_x+1, 1))
	plt.yticks(np.arange(0, 1.1, 0.1)) 

	#ax[1].set_xlabel('number of attractors')
	#print(correct/float(count))
	print(model)
	print(acc_count_attractor)
	#print(acc_count_attractor_single_entity)
	index+=1
plt.legend(loc='upper right', bbox_to_anchor=(1, -0.08),fancybox=True, shadow=True, ncol=8, borderpad=0.09)
#plt.legend(loc='upper right', bbox_to_anchor=(1.1, -0.08),fancybox=True, shadow=True, ncol=8)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fancybox=True, shadow=True, ncol=1)
plt.show()
#plt.savefig('multiple_entity/plots/NoAttractorsSingleandMultipleVsAcc.png',bbox_inches = 'tight')


