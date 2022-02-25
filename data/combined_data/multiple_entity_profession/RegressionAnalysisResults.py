import numpy as np
import csv
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
#from feature_engine.discretisers import EqualWidthDiscretiser
from yellowbrick.regressor import ResidualsPlot
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import itertools
from scipy import stats
import statsmodels as statmodel
from sklearn.decomposition import PCA


req_plot = False
statsmodel = True

########################types of probability########################################
scale = 1 #0: standard data is used, 1: if we take natural log of probability, 2: if we take negative logarithm of target probability


model = 'BertBase' #'GPT2Small'

dataset = open('complete_data_'+model+'.csv')
#dataset = open('datasets/BackUpData/complete_data_'+model+'.csv')

reader = csv.DictReader(dataset, delimiter='\t')
dir = './'
#dir = 'multiple_entity/plots/'+model+'/'
if scale==0:
	resultFile = open(dir+model+'Results.txt','w')
if scale==1:
	resultFile = open(dir+model+'ResultsWithLog.txt','w')
if scale==2:
	resultFile = open(dir+model+'ResultsWithNegativeLog.txt','w')
def standardizeInput(inputdata):
	scaler = StandardScaler()
	standard_data = scaler.fit_transform(inputdata)
	return standard_data

data = []
target = []
a = []
b = []

for row in reader:
	instance = []
	#if int(row['target_indicator']) == 1:

	instance.append(int(row['id']))
	instance.append(int(row['distance_from_beginning']))
	instance.append(int(row['distance_from_masked_token']))
	instance.append(int(row['target_index']))
	#changing probability to natural log
	if scale==0:
		instance.append(float(row['target_prob_without_attractor']))
	if scale==1:
		instance.append(np.log(float(row['target_prob_without_attractor'])))
	if scale==2:
		instance.append(-np.log(float(row['target_prob_without_attractor'])))

	#changing probability to natural log
	if scale==0:
		instance.append(float(row['max_remaining_prob']))
	if scale==1:
		instance.append(np.log(float(row['max_remaining_prob'])))
	if scale==2:
		instance.append(-np.log(float(row['max_remaining_prob'])))
	#changing probability to natural log
	if scale==0:
		instance.append(float(row['avg_remaining_prob']))
	if scale==1:
		#b.append(float(row['avg_remaining_prob']))
		instance.append(np.log(float(row['avg_remaining_prob'])))
	if scale==2:
		instance.append(-np.log(float(row['avg_remaining_prob'])))

	instance.append(float(row['target_indicator']))
	#changing probability to natural log
	if scale==0:
		instance.append(float(row['target_prob']))
	if scale==1:
		instance.append(np.log(float(row['target_prob'])))
	if scale==2:
		instance.append(-np.log(float(row['target_prob'])))
	data.append(instance)


data = np.asarray(data)

target = data[:,-1]
target = np.reshape(target, (target.shape[0],1))


data_index = data[:,0]

data_index = np.reshape(data_index, (data_index.shape[0],1))

standardized_data_only = standardizeInput(data[:,1:-1])

target = np.reshape(target,(target.shape[0],1))
standardized_data = np.hstack((data_index, standardized_data_only))

standardized_data=np.hstack((standardized_data,target))

normal_column = standardized_data[:,1]

data = np.random.permutation(standardized_data)



#################################################data analysis for linearity#################################################

column_names = ['distance_from_beginning','distance_from_masked_token','target_index','target_prob_without_attractor'\
		,'max_remaining_prob','target_indicator']

ex_data_df = pd.DataFrame(data[:,[1,2,3,4,5,7]], columns=column_names)
ex_data_df['target_prob'] = data[:,8]


formula_str = ex_data_df.columns[-1]+' ~ '+'+'.join(ex_data_df.columns[:-1])
print('formula_str ',formula_str)
resultFile.write('variance_inflation_factor')
resultFile.write('\n')
for i in range(0, len(ex_data_df.columns[:-1])):	
	v = vif(np.matrix(ex_data_df[:-1]),i)
	resultFile.write('vif '+str(ex_data_df.columns[i])+' '+str(round(v,2)))
	resultFile.write('\n')

modelOLS=sm.ols(formula=formula_str, data=ex_data_df)
fitted = modelOLS.fit()
print(fitted)
resultFile.write("***********************Fitted OLS summary***************")
resultFile.write('\n')
resultFile.write(fitted.summary().as_text())
resultFile.write('\n')




df_result=pd.DataFrame()

df_result['pvalues']=fitted.pvalues[1:]
df_result['Features']=ex_data_df.columns[:-1]
df_result.set_index('Features',inplace=True)

resultFile.write(df_result.to_string(header = True, index = False))
resultFile.write('\n')
column_names = ['distance_from_beginning','distance_from_masked_token','target_index','target_prob_without_attractor'\
		,'max_remaining_prob']
ex_data_df = pd.DataFrame(data[:,[1,2,3,4,5]], columns=column_names)
ex_data_df['target_prob'] = data[:,8]


formula_str = ex_data_df.columns[-1]+' ~ '+'+'.join(ex_data_df.columns[:-1])
print('formula_str ',formula_str)
resultFile.write('variance_inflation_factor')
resultFile.write('\n')
for i in range(0, len(ex_data_df.columns[:-1])):	
	v = vif(np.matrix(ex_data_df[:-1]),i)
	resultFile.write('vif '+str(ex_data_df.columns[i])+' '+str(round(v,2)))
	resultFile.write('\n')

modelOLS=sm.ols(formula=formula_str, data=ex_data_df)
fittedWithout = modelOLS.fit()
print(fittedWithout)
resultFile.write("***********************Fitted OLS summary***************")
resultFile.write('\n')
resultFile.write(fitted.summary().as_text())
resultFile.write('\n')
likelihood_ratio, p_value, df = fitted.compare_lr_test(fittedWithout)
#likelihood_ratio, p_value, df = statmodel.regression.linear_model.OLSResults.compare_lr_test(fitted, fittedWithout)
print('likelihood ratio test ',likelihood_ratio, p_value, df)

########################################### mixed regression model ################################
'''md = smf.mixedlm(formula = formula_str, data=ex_data_df, groups=ex_data_df["distance_from_masked_token"])
mdf = md.fit()
print(mdf.summary())
resultFile.write("***********************Fitted Mixed regression model summary***************")
resultFile.write('\n')
resultFile.write(mdf.summary().as_text())
resultFile.write('\n')'''
print('***************spearmanr************************')
spearman_corr ,p_value=stats.spearmanr(np.asarray(data[:,1]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.spearmanr(np.asarray(data[:,2]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.spearmanr(np.asarray(data[:,3]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.spearmanr(np.asarray(data[:,4]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.spearmanr(np.asarray(data[:,5]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.spearmanr(np.asarray(data[:,7]), data[:,8])
print(spearman_corr ,' ',p_value)
print('***************pointbiserialr************************')
spearman_corr ,p_value=stats.pointbiserialr(np.asarray(data[:,1]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pointbiserialr(np.asarray(data[:,2]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pointbiserialr(np.asarray(data[:,3]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pointbiserialr(np.asarray(data[:,4]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pointbiserialr(np.asarray(data[:,5]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pointbiserialr(np.asarray(data[:,7]), data[:,8])
print(spearman_corr ,' ',p_value)
print('***************pearson************************')
spearman_corr ,p_value=stats.pearsonr(np.asarray(data[:,1]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pearsonr(np.asarray(data[:,2]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pearsonr(np.asarray(data[:,3]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pearsonr(np.asarray(data[:,4]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pearsonr(np.asarray(data[:,5]), data[:,8])
print(spearman_corr ,' ',p_value)
spearman_corr ,p_value=stats.pearsonr(np.asarray(data[:,7]), data[:,8])
print(spearman_corr ,' ',p_value)


if req_plot==True:
	plt.rcParams["figure.figsize"] = [16,4.8]
	target = np.asarray(data[:,-1])
	ex_data = np.asarray(data[:,0:-1])
	ex_data = np.asarray(data[:,[1,2,3,4,5,7]])
	print(ex_data[0])
	column_1 = np.asarray(data[:,1])
	column_2 = np.asarray(data[:,2])
	column_3 = np.asarray(data[:,3])
	column_4 = np.asarray(data[:,4])
	column_5 = np.asarray(data[:,5])
	column_6 = np.asarray(data[:,7])
	
	###############################################################################################################################
    

	######################################################Generating pairplot######################################################
	plt.figure(figsize=(18,10))

	
	pearson_corr = ex_data_df.corr()
	sns.pairplot(pearson_corr)
	if scale==0:
		plt.savefig(dir+'pairwisePearsonPlot'+model+'.png')
	if scale==1:
		plt.savefig(dir+'pairwisePearsonPlotWithLog'+model+'.png')
	if scale==2:
		plt.savefig(dir+'pairwisePearsonPlotWithNegativeLog'+model+'.png')
	plt.close()
	#############################################################Generating heatmap###################################################
	plt.figure(figsize=(18,10))
	sns.heatmap(pearson_corr,annot=True,linewidths=2)
	if scale==0:
		plt.savefig(dir+'heatmapPearsonPlot'+model+'.png')
	if scale==1:
		plt.savefig(dir+'heatmapPearsonPlotWithLog'+model+'.png')
	if scale==2:
		plt.savefig(dir+'heatmapPearsonPlotWithNegativeLog'+model+'.png')
	plt.close()
	########################################################################################################################
	######################################################Generating pairplot######################################################
	plt.figure(figsize=(18,10))

	
	spearman_corr = ex_data_df.corr('spearman')
	sns.pairplot(spearman_corr)
	if scale==0:
		plt.savefig(dir+'pairwiseSpearmanPlot'+model+'.png')
	if scale==1:
		plt.savefig(dir+'pairwiseSpearmanPlotWithLog'+model+'.png')
	if scale==2:
		plt.savefig(dir+'pairwiseSpearmanPlotWithNegativeLog'+model+'.png')
	plt.close()
	#############################################################Generating heatmap###################################################
	plt.figure(figsize=(18,10))
	sns.heatmap(spearman_corr,annot=True,linewidths=2)
	if scale==0:
		plt.savefig(dir+'heatmapSpearmaPlot'+model+'.png')
	if scale==1:
		plt.savefig(dir+'heatmapSpearmaPlotWithLog'+model+'.png')
	if scale==2:
		plt.savefig(dir+'heatmapSpearmaPlotWithNegativeLog'+model+'.png')
	plt.close()
	########################################################################################################################

	############################################Generating scatter plot###########################################################
	fig = plt.figure(figsize=(18,10))

	plt.scatter(column_1, target, marker='x',color='b',label='distance_from_beginning')

	plt.title("distance_from_beginning vs. target_prob ", fontdict={'fontsize':10})
	plt.xlabel('distance_from_beginning')
	plt.ylabel('target_prob')
	plt.savefig(dir+'scatter_plot_distance_from_beginning_vs_target_prob.png')
	plt.close()

	plt.scatter(column_2, target, marker='x',color='b',label='distance_from_masked_token')

	plt.title(" distance_from_masked_token vs. target_prob", fontdict={'fontsize':10})
	plt.xlabel('distance_from_masked_token')
	plt.ylabel('target_prob')
	plt.savefig(dir+'scatter_plot_distance_from_masked_token_vs_target_prob.png')
	plt.close()

	plt.scatter(column_3, target, marker='x',color='b',label='target_index')
	plt.title(" target_index vs. target_prob", fontdict={'fontsize':10})
	plt.xlabel('target_index')
	plt.ylabel('target_prob')
	plt.savefig(dir+'scatter_plot_target_index_vs_target_prob.png')
	plt.close()

	plt.scatter(column_4, target, marker='x',color='b',label='target_prob_without_attractor')
	plt.title(" target_prob_without_attractor vs. target_prob", fontdict={'fontsize':10})
	plt.xlabel('target_prob_without_attractor')
	plt.ylabel('target_prob')
	plt.savefig(dir+'scatter_plot_target_prob_without_attractor_vs_target_prob.png')
	plt.close()

	plt.scatter(column_5, target, marker='x',color='b',label='max_remaining_prob')
	plt.title(" max_remaining_prob vs. target_prob", fontdict={'fontsize':10})
	plt.xlabel('max_remaining_prob')
	plt.ylabel('target_prob')
	plt.savefig(dir+'scatter_plot_max_remaining_prob_vs_target_prob.png')
	plt.close()


	plt.scatter(column_6, target, marker='x',color='b',label='target_indicator')
	plt.title("target_indicator  vs. target_prob", fontdict={'fontsize':10})
	plt.xlabel('target_indicator')
	plt.ylabel('target_prob')

	plt.savefig(dir+'scatter_plot_target_indicator_vs_target_prob.png')
	plt.close()

#############################################################################################################################
print('************************************ analysis of regression results ***********************************')

train_data = data[0:int(data.shape[0]*0.75),:]   # select 75% for train data
test_data = data[int(data.shape[0]*0.75):,:]

train_target = np.asarray(train_data[:,-1])

train_data = np.asarray(train_data[:,0:-1])
#train_data = np.asarray(train_data[:,0])
test_target = np.asarray(test_data[:,-1])

test_data = np.asarray(test_data[:,0:-1])



def calculateLinearRegression(cols):

	#clf = LinearRegression()
	clf = Ridge(alpha=0.3)

	train_data_columns = train_data[:,cols]
	test_data_columns = test_data[:,cols]
	clf.fit(train_data_columns, train_target)
	test_data_columns = test_data[:,cols]
	test_index = [i for i in range(0,test_target.shape[0])]
	test_prediction = clf.predict(test_data_columns)
	r2_score = metrics.r2_score(test_target, test_prediction)
	print(test_data_columns.shape)
	deno = test_data_columns.shape[0]-test_data_columns.shape[1]-1
	nume = test_data_columns.shape[0]-1
	adjustedR2_score = 1-(1-r2_score)*(nume/deno)
	mean_absolute_error = metrics.mean_absolute_error(test_target,test_prediction)
	mean_squared_error = metrics.mean_squared_error(test_target,test_prediction)
	RMSE = np.sqrt(metrics.mean_squared_error(test_target,test_prediction))

	return r2_score, mean_absolute_error, mean_squared_error, RMSE, adjustedR2_score

data_columns = ['distance_from_beginning','distance_from_masked_token','target_index','target_prob_without_attractor'\
	,'max_remaining_prob','target_indicator']

data_columns = [1,2,3,4,5,7]
#data_columns = [1,2,3,4,5,6]
R2ScoreList = []
feature_list = []
num_features = []
adjustedR2_scoreList = []

#standard_train_data = scaler.fit_transform(train_data[:,1:])
for k in range(1,len(data_columns) + 1):
	for combo in itertools.combinations(data_columns,k):
		print(combo)
		r2_score, mean_absolute_error, mean_squared_error, RMSE, adjustedR2_score = calculateLinearRegression(list(combo))
		R2ScoreList.append(r2_score)
		feature_list.append(combo)
		num_features.append(len(combo))
		adjustedR2_scoreList.append(adjustedR2_score)
		#print('r2_score ',r2_score, 'mean_absolute_error ', mean_absolute_error, ' mean_squared_error ', mean_squared_error, ' RMSE ',RMSE)

		#input()
df = pd.DataFrame({'num_features': num_features, 'R_squared':R2ScoreList,'features':feature_list,'adjustedR2_score':adjustedR2_scoreList})
df_max = df[df.groupby('num_features')['R_squared'].transform(max) == df['R_squared']]
df['max_R_squared'] = df.groupby('num_features')['R_squared'].transform(max)
resultFile.write(df.to_string(header = True, index = False))
resultFile.write('\n')

#########################################3dimensionality reduction ###################################################

cols = [1,2,3,4,5,7]
#train_data_columns = train_data[:,cols]

train_data_columns = data[:,cols]
train_target = data[:,8]


test_data_columns = test_data[:,cols]
clf = LinearRegression()
clf.fit(train_data_columns, train_target)
resultFile.write('clf coeeficients '+str(clf.coef_))
resultFile.write('\n')
cdf = pd.DataFrame(data=clf.coef_, index=data_columns, columns=["Coefficients"])
#######################################################calculate standard errors and t-statistic for the coefficients##################
n=train_data_columns.shape[0]
k=train_data_columns.shape[1]

dfN = n-k
train_pred=clf.predict(train_data_columns)
print('train_data ',train_data_columns)
#train_pred = clf.decision_function(train_data_columns)
#train_pred = (train_pred - train_pred.min()) / (train_pred.max() - train_pred.min())

train_residuals = train_pred - train_target
train_error = np.square(train_pred - train_target)
plt.show()
sum_error=np.sum(train_error)
se=[0,0,0,0,0,0,0]
se = [0,0,0,0,0,0]

for i in range(k):
    r = (sum_error/dfN)
    r = r/np.sum(np.square(train_data_columns[i]-train_data_columns[i].mean()))
    se[i]=np.sqrt(r)

sse = np.sum((train_pred - train_target) ** 2, axis=0) / float(n - k)
se1 = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(train_data_columns.T, train_data_columns))))])
print(se1)
print(type(sse))
print(sse.shape)
print(type(se))
t_value= clf.coef_/se1[0]
print(t_value)
cdf['sse'] = sse
cdf['se1'] = se1[0]
cdf['Standard Error']=se
#cdf['t-statistic']=cdf['Coefficients']/cdf['Standard Error']
cdf['t-statistic']=cdf['Coefficients']/cdf['se1']
cdf["p-value"] = np.squeeze(2 * (1 - stats.t.cdf(np.abs(t_value), train_target.shape[0] - k)))
resultFile.write('\n')
resultFile.write("R2-squared value of train predictions:"+str(metrics.r2_score(train_target, train_pred)))
resultFile.write('\n')
#print("R2-squared value of train predictions:",metrics.r2_score(train_target, train_pred))
resultFile.write(cdf.to_string(index=False))
resultFile.write('\n')
#print(cdf)
visualizer = ResidualsPlot(model = clf)
visualizer.fit(train_data_columns, train_target)
visualizer.score(test_data_columns, test_target)
visualizer.show()

#####################################################################################


test_prediction = clf.predict(test_data_columns)



temp_r2_score_result_with_standardizing = metrics.r2_score(test_target, test_prediction)
resultFile.write("Mean absolute error (MAE): "+str(metrics.mean_absolute_error(test_target,test_prediction))+ "\n")
resultFile.write("Mean square error (MSE):" +str(metrics.mean_squared_error(test_target,test_prediction))+"\n")
resultFile.write("Root mean square error (RMSE):"+str(np.sqrt(metrics.mean_squared_error(test_target,test_prediction)))+"\n")
resultFile.write("R2-squared value of test predictions:"+str(temp_r2_score_result_with_standardizing)+"\n")





	