#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:58:31 2020

@author: vincenthall
company: Build Intellect Limited
"""

##########################################################
# 1. Define the problem: predict heart failure, what data do we have?
#111111111111111111111111111111111111111111111111111111111
# 4 datasets, all processed, heart disease, collected from 1990 to 1996.
#Data Set Information:
#
#This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
#this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).
#
#The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.
#
#One file has been "processed", that one containing the Cleveland database. All four unprocessed files also exist in this directory.
#
#To see Test Costs (donated by Peter Turney), please see the folder "Costs"
#
#
#Attribute Information:
#
#Only 14 attributes used:
#0. #3 (age)
#1. #4 (sex)
#2. #9 (cp)
#3. #10 (trestbps)
#4. #12 (chol)
#5. #16 (fbs)
#6. #19 (restecg)
#7. #32 (thalach)
#8. #38 (exang)
#9. #40 (oldpeak)
#10. #41 (slope)
#11. #44 (ca)
#12. #51 (thal)
#13. #58 (num) (the predicted attribute) 0 = no presence, 4 is max presence of heart disease.

########################################
# 2. Load the data, then get to know it
#22222222222222222222222222222222222222
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,SCORERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict,StratifiedKFold






dataFolder = "/Users/vincenthall/Data/HeartFailure/"

def loadHFData(dataFolder,fileName):
    # Oh, the data is just csv.
    outData = pd.read_csv(dataFolder+fileName) 
    return outData

fileName = "processed.cleveland.data"
cleveland = loadHFData(dataFolder,fileName)

fileName = "processed.hungarian.data"
hungarian = loadHFData(dataFolder,fileName)

fileName = "processed.switzerland.data"
switzerland = loadHFData(dataFolder,fileName)

fileName = "processed.va.data"
va = loadHFData(dataFolder,fileName)

# Get to know the data
print(cleveland.head())
# Tiny bit of missing data.

print(hungarian.head()) 
#there is some missing data here.

print(switzerland.head())
# More missing data.

print(va.head()) # Long Beach, California.
# More missing data: "?"

# Beginning to see why only Cleveland data used in other people's work.

# I don't want to mess with data, so I just delete rows with missing data.
# Let's prioritise cleveland data.

# There are some missing values, represented by "?" So remove the rows.
clevelandReplaced = cleveland.replace(to_replace="?", value=np.nan)
clevelandReplaced.dropna(axis=0, inplace=True) # axis=0 means delete rows

# Count remaining nans:
print(pd.DataFrame(clevelandReplaced).isnull().sum()) # Yeah! Got those nans!

# For extra data, I could try combining all datasets, then removing all nans or "?"s.
# Maybe as another version.


cleveDescribe = clevelandReplaced.describe
print(cleveDescribe)
""" 
mean and quintiles: min, 25%, 50%, 75%, max.
largest mean is 246.7, smallest mean is 0.1457, clearly some scaling 
will have to be done to avoid the ML method treating the 
large-valued column as more important.
"""

####################################################
# 3. Reshape the data for the prediction algorithms?
#333333333333333333333333333333333333333333333333333

# There are some hidden string entries.
clevelandReplaced.dtypes
# So two columns are objects.

clevelandReplaced['0.0.1'] = clevelandReplaced['0.0.1'].astype(float)
clevelandReplaced['6.0'] = clevelandReplaced['6.0'].astype(float)

# How about now?
clevelandReplaced.dtypes
# All floats, except the label, which I need to take out.

# First, do some scaling because the column/variable value sizes are all different.
# z-score is good.
start = 0  # First input data column.
fin = 13  # Final input data column. actually leaves out column 13, which is the heart disease num.   
clevelandScaled = stats.zscore(clevelandReplaced.iloc[:, start:fin].values, axis=0, ddof=1)
# axis=0 does it down the columns, so each column is mean=0, stdev=1.

labels = clevelandReplaced["0"] # Do NOT scale these labels.

# Now it's numpy array, so convert to pandas.
clevelandScaledPD = pd.DataFrame(data=clevelandScaled, index=clevelandReplaced.index.values, columns=clevelandReplaced.columns.values[0:-1])   

# It's a use of memory to keep all these versions of cleveland, but it does avoid 
# confusion and executing code in the wrong order.

cleveScaledDisc = clevelandScaledPD.describe() 
# mean always <= 10^-16, std always 1.0, great!

# Is the data skewed?
clevelandScaledPD.skew()
# skew<-1 or >1 is highly skewed.
# So print those names.
these = np.where(abs(clevelandScaledPD.skew()) > 1) 
print("These columns are skewed:")
print(clevelandScaledPD.columns.values[these])
print("Columns by number:")
print(these)

# I'm going to try remove the skewnesses of these, all columns:
minMax = MinMaxScaler()
minMax.fit(clevelandScaledPD)
cleveNorm = minMax.transform(clevelandScaledPD)

# Check if that was a success:

# But it must be pd.DataFrame to get the skew
cleveNormPD = pd.DataFrame(data=cleveNorm, index=clevelandScaledPD.index.values, columns=clevelandScaledPD.columns.values)   
cleveNormPD.skew()

these = np.where(abs(cleveNormPD.skew()) > 1) 
print("These columns are skewed:")
print(cleveNormPD.columns.values[these])
# That was not a success, and normalising it before scaling didn't help.
# I need to move on to the prediction part.


# Feature selection
# Back to the original dataset:
print("Let's visualise with histograms to see if there is anything obvious.")
names = list(cleveland.columns.values)
names = names[0:-1]
plt.rcParams["figure.figsize"] = (3,3)

for i in range(len(names)):
  plt.figure()
  plt.rcParams.update({'font.size':15})
  plt.title(names[i])
  cleveland.groupby(('0'))[names[i]].hist(alpha=0.68)
  plt.rcParams.update({'font.size':11})
  plt.gca().legend(('0','1','2','3','4')) # I checked with np.where(dataset.diagnosis=='M')[0] for this labelling, below.

plt.show()
print("2.3, or column 10 is not terrible at differentiating the classes 0 to 4, not good though.")

print("Train test split") # Not doing a validation set, to get this done more quickly.
print("No time to do multiclass correctly, so I'm going to take the step to make it binary.")
print("So any label above 1 is called '1' and 0s and 1s become zeros. ")

labels1 = labels
labels = np.array(labels)
labels[labels < 2] = 0
labels[labels > 1] = 1

len(labels1[labels1 < 2])

len(labels[labels < 1])
# same number, good, then I must have done this right.
len(labels1[labels1 > 1])
len(labels[labels == 1 ])
# Again, same number, good.
del labels1

#train / validation spilt
validation_size = 0.4
# Use stratify to keep the class imbalance. I'll assume it worked, for time management.
X_train, X_test, Y_train, Y_test = train_test_split(cleveNormPD, labels, test_size = validation_size, random_state = 0, stratify=labels)

trainCount = np.shape(X_train)[0]
testCount = np.shape(X_test)[0]
print("Num training samples is "+str(trainCount))
print("Num test samples is "+str(testCount))

print("which means that test samples are" +str((testCount/(testCount+trainCount))*100) +"% of the total data size.")
print("About 40%, good.")

print("Use extra trees to select features more automatically:")
# fit an Extra Trees model to the data
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X_train,Y_train)
#model.fit(X_train,Y_train.reshape(Y_train.shape[0]))


plt.plot(model.feature_importances_)
plt.title("Importances of features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

ind = np.argsort(model.feature_importances_,axis=0)
print("Features least important..")
print(ind)
print("to most important")
print("Huh, I thought 10 would be a lot higher importance.")
print("\n")
print("These are the feature importances: \n"+str(model.feature_importances_))
print("\n")
# Of the 13, let's select the top 6, say
index = ind[-6:] # Of the feature importance incides, take the last 6, a.k.a. the most important.
X_Red = X_train.iloc[:,index]


###################################################
# 4. Compare prediction algorithms
#44444444444444444444444444444444444444444444444444

print("Let's do model selection.")


# prepare ML models
models = []
models.append(('LR', LogisticRegression(solver = 'lbfgs',max_iter=200,class_weight='balanced', multi_class="multinomial")))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA',QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('NB', GaussianNB()))
models.append(('SVMRBF', SVC(kernel='rbf',gamma='auto',class_weight='balanced')))
models.append(('SVM', SVC(kernel='linear',gamma='auto',class_weight='balanced')))
models.append(('MLP',MLPClassifier(hidden_layer_sizes = 15, alpha=1, max_iter=750)))

seed = 9

print("Compare accuracies and areas under the ROC curves (roc_auc)")
print("Though there is a class imbalance, so I trust the ROC_AUC more.")
for i in range(2):
# evaluate each model in turn
    results = []
    names = []
    if i == 0:
      scoringmetric = 'roc_auc'
    else:
      scoringmetric = 'accuracy'
    
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model,X_train , Y_train, cv=kfold, scoring=scoringmetric)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,10)
    fig.suptitle('Algorithm Comparison' + str ('Metric: ') + str(scoringmetric))
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
print("Select the favoured model based on the bottom of the box in the box and whiskers plot.")
print("That takes care of the worse case scenario.")

print("On ROC_AUC: SVM RBF (support vector machine, with radial basis function) has the highest bottom of box, and is very tight.")
print("And on Accuracy, SVMRBF is the second best")
print("'SVM', which has linear kernel, is the inverse, best on Acc and 2nd on ROC_AUC.")
###################################################
# 5. Improve selected prediction algorithm
#55555555555555555555555555555555555555555555555555


#create a pipeline
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('pca', PCA()))
estimators.append(('SVMRBF',SVC(kernel='rbf',C=1.0,gamma='auto',class_weight='balanced', probability=True)))


#we need to determine the number of features that we should use from PCA
#we need to determine the regularisation parameter, C in SVM classifier to avoid overfitting
param_grid = {'pca__n_components': [2,3,4,5,6,7,8,9,10]}

modelHPS = Pipeline(estimators) # evaluate pipeline

#HyperParameter Selection
search = GridSearchCV(modelHPS, param_grid, iid=False, cv=10,return_train_score=False,scoring='roc_auc')
search.fit(X_train,Y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# Results
n_repeats = 10
best_pipeline = search.best_estimator_
threshold = 0.1 # could try 0.001, 0.3 and see how the metrics change: sens, spec, NPV, AUC.

auc_cv = np.zeros(n_repeats)
sen_cv = np.zeros(n_repeats)
spe_cv = np.zeros(n_repeats)
npv_cv = np.zeros(n_repeats)

for i in range(n_repeats):
    y_pred_cv = np.zeros(Y_train.shape)
    y_proba_cv  = cross_val_predict(best_pipeline,X_train,Y_train,cv=StratifiedKFold(3, shuffle=True),method = "predict_proba")
    y_proba_cv = y_proba_cv[:,1]
    y_pred_cv[y_proba_cv > threshold] = 1.0
    fpr_cv, tpr_cv, thresholds_cv = roc_curve(Y_train, y_proba_cv)
    auc_cv[i] = roc_auc_score(Y_train, y_proba_cv)
    tn_cv,fp_cv,fn_cv,tp_cv = confusion_matrix(Y_train,y_pred_cv).ravel()
    sen_cv[i] = tp_cv / (tp_cv + fn_cv)
    spe_cv[i] = tn_cv / (tn_cv + fp_cv)
    npv_cv[i] = tn_cv / (tn_cv + fn_cv)
    
print("sensitivity %0.3f (+/- %0.01g)" % (sen_cv.mean(),1.96 * sen_cv.std() / np.sqrt(n_repeats)))
print("specificity %0.3f (+/- %0.01g)" % (spe_cv.mean(),1.96 * spe_cv.std() / np.sqrt(n_repeats)))
print("NPV %0.3f (+/- %0.01g)" % (npv_cv.mean(),1.96 * npv_cv.std() / np.sqrt(n_repeats)))
print("AUC %0.4f (+/- %0.01g)" % (auc_cv.mean(),1.96 * auc_cv.std() / np.sqrt(n_repeats)))
# using %0.01g rounds to 1 significant figure.
# using %0.3f rounds to 3 decimal places

###################################################
# 6. Present results
#66666666666666666666666666666666666666666666666666
# Use the test data.
# Present Results: (build using training data only and apply to test data)
# prepare final model and apply to test data


#finalise the model
preprocess_pipeline = make_pipeline(StandardScaler(), PCA(n_components=search.best_params_['pca__n_components']))
#prepare the data (train and validation)
preprocess_pipeline.fit(X_train)
X_train_transformed = preprocess_pipeline.transform(X_train)
X_test_transformed = preprocess_pipeline.transform(X_test)
#fit to the whole data
model = SVC(kernel='rbf',C=1.0,gamma='auto',class_weight='balanced', probability=True)
model.fit(X_train_transformed,Y_train)
predictions = model.predict(X_test_transformed)
probs = model.predict_proba(X_test_transformed)
probsx = probs[:,1]
print('Best Model Parameters and fitting to the final data')
print('intercept is',model.intercept_)


# Sensitivity_and_specificity assuming a threshold on test data

fpr, tpr, thresholds = roc_curve(Y_test, probsx)
threshold = 0.1
y_pred = np.zeros(probsx.shape)
y_pred[probsx > threshold] = 1
tn,fp,fn,tp = confusion_matrix(Y_test,y_pred).ravel()
sen = tp / (tp+fn)
spec = tn / (tn+fp)
npv = tn / (tn + fn)
### calculate AUC
auc = roc_auc_score(Y_test, probsx)
print("AUC {}". format(auc))
print("Cut-off {}". format(threshold))
print("NPV: %0.3f" %(npv))
print("Sensitivity: %0.3f" %(sen))
print("Specificity: %0.3f" %(spec))


###Plot and show ROC
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='*')
# show the plot
plt.xlabelauc = roc_auc_score(Y_test, probsx)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC - Heart Disease')
plt.show()


## Present confusion matrix
#check the results
print('The accuracy score is')
print(accuracy_score(Y_test, predictions))
print('The confusion matrix is')
print(confusion_matrix(Y_test, predictions))
print('The classification report is')
print(classification_report(Y_test, predictions))
