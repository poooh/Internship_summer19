import sys
import os
from txtTocsv import TextToCsv
from Model import model_utils
import pandas as pd
import grid_search
import warnings
from sklearn.externals import joblib

warnings.filterwarnings("ignore")

def getargs():
	data_directory = sys.argv[1]
	return data_directory

if __name__ == "__main__":
	data_directory = getargs()
	tc = TextToCsv()
	# textdata = tc.getdata(data_directory)
	textdata = tc.getdirdata(data_directory)
	dir_path = os.path.dirname(os.path.realpath(__file__))
	csvdata = tc.getcsv(textdata, dir_path)
	df = pd.read_csv(csvdata)
	fit_model, tf = model_utils.get_best_model(df)
	joblib.dump(tf, 'tfidfVectorizer.pkl')
	joblib.dump(fit_model, 'classifier.pkl')
	joblib.dump(tf, '../api/tfidfVectorizer.pkl')
	joblib.dump(fit_model, '../api/classifier.pkl')

	# trainmatrix, testmatrix, tfidf_trainmatrix, tfidf_testmatrix, fit_model, fv, tf = model_utils.get_best_model_grid(df)
	# grid_search.gridsearch(tfidf_trainmatrix,trainmatrix, fit_model)