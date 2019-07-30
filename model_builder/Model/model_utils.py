import sys
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from Model import model_utils, feature_engineer, basemodel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

#this splits the whole dataset into train and test
def split_train_test(df):
	datatypeval = ['train', 'test']
	df['datatype'] = pd.Series(np.random.choice(datatypeval,4889, p=[.8,.2]), index=df.index)
	trainmatrix = df[(df.datatype == 'train')]
	testmatrix = df[(df.datatype == 'test')]
	print(len(trainmatrix))
	print(len(testmatrix))
	return trainmatrix, testmatrix

#This makes or fits the model for given classifier on given dataset
def makemodel(clf, vectorized_trainmatrix, trainmatrix):
	vectorized_trainmatrix = vectorized_trainmatrix.toarray()
	clf.fit(vectorized_trainmatrix, trainmatrix['Classname'])
	return clf

#This gives the pridiction for the test dataset.
def getprediction(clf, vectorized_testmatrix):
	vectorized_testmatrix =  vectorized_testmatrix.toarray()
	predicted = clf.predict(vectorized_testmatrix)
	return predicted

#This calculates the accuracy for prediction.
def getaccuracy(testmatrix, predicted):
	acc = accuracy_score(testmatrix['Classname'], predicted)
	return acc


def get_best_model(df):
 	kf = KFold(n_splits = 10, shuffle = True, random_state = 12)
 	base_clf = basemodel.Classifiers()
 	svm = base_clf.linearsvc()
 	clf = CalibratedClassifierCV(svm)
 	# clf = base_clf.svm()
 	# clf = base_clf.linearsvc()
 	# clf = base_clf.linearclassifier()
 	max = 0.000
 	fv = feature_engineer.FeatureVectorizor()
 	trainmatrix = pd.DataFrame()
 	testmatrix = pd.DataFrame()
 	for x in kf.split(df):
 		result = x
 		x_trainmatrix = df.iloc[result[0]]
 		x_testmatrix =  df.iloc[result[1]]
 		x_trainmatrix = x_trainmatrix.fillna(' ')
 		x_testmatrix = x_testmatrix.fillna(' ')
 		tfidf_trainmatrix, tfidf_testmatrix, tf = fv.get_tfidf_vectors(x_trainmatrix, x_testmatrix)
 		strng = model_utils.makemodel(clf, tfidf_trainmatrix, x_trainmatrix)
 		predicted = model_utils.getprediction(clf, tfidf_testmatrix)
 		acc = model_utils.getaccuracy(x_testmatrix, predicted)
 		print(acc)
 		if acc > max:
 			max = acc
 			trainmatrix = x_trainmatrix
 			testmatrix =  x_testmatrix
 	tfidf_trainmatrix, tfidf_testmatrix, tf = fv.get_tfidf_vectors(trainmatrix, testmatrix)
 	fit_clf = model_utils.makemodel(clf, tfidf_trainmatrix, trainmatrix)
 	print("model fit")
 	predicted = model_utils.getprediction(fit_clf, tfidf_testmatrix)
 	acc = model_utils.getaccuracy(testmatrix, predicted)
 	print(acc)
 	return fit_clf, tf

def get_best_model_grid(df):
 	kf = KFold(n_splits = 10, shuffle = True, random_state = 12)
 	base_clf = basemodel.Classifiers()
 	clf = base_clf.linearsvc()
 	max = 0.000
 	fv = feature_engineer.FeatureVectorizor()
 	trainmatrix = pd.DataFrame()
 	testmatrix = pd.DataFrame()
 	for x in kf.split(df):
 		result = x
 		x_trainmatrix = df.iloc[result[0]]
 		x_testmatrix =  df.iloc[result[1]]
 		tfidf_trainmatrix, tfidf_testmatrix, tf = fv.get_tfidf_vectors(x_trainmatrix, x_testmatrix)
 		strng = model_utils.makemodel(clf, tfidf_trainmatrix, x_trainmatrix)
 		predicted = model_utils.getprediction(clf, tfidf_testmatrix)
 		acc = model_utils.getaccuracy(x_testmatrix, predicted)
 		if acc > max:
 			max = acc
 			trainmatrix = x_trainmatrix
 			testmatrix =  x_testmatrix
 	tfidf_trainmatrix, tfidf_testmatrix, tf = fv.get_tfidf_vectors(trainmatrix, testmatrix)
 	fit_clf = model_utils.makemodel(clf, tfidf_trainmatrix, trainmatrix)
 	print("model fit")
 	predicted = model_utils.getprediction(fit_clf, tfidf_testmatrix)
 	acc = model_utils.getaccuracy(testmatrix, predicted)
 	print(acc)
 	return trainmatrix, testmatrix, tfidf_trainmatrix, tfidf_testmatrix, fit_clf, fv, tf