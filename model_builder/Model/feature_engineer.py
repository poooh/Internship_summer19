from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score

#Feature Engineering Methodologes
class FeatureVectorizor(object):

	#This extracts the features and get the Tf - Idf score for it.
	def get_tfidf_vectors(self, trainmatrix, testmatrix):
		tf = TfidfVectorizer(analyzer='word', stop_words = 'english',smooth_idf=True,use_idf=True, max_features=1000)
		# tf = TfidfVectorizer(analyzer='word', stop_words = 'english',smooth_idf=True,use_idf=True)
		# trainmatrix.replace(np.nan, '', regex=True)
		# trainmatrix.fillna(' ')
		# testmatrix.fillna(' ')
		tfidf_trainmatrix =  tf.fit_transform(trainmatrix['Content'])
		# tfidf_trainmatrix =  tf.fit_transform(trainmatrix['Content'].values.astype('U'))
		print(tfidf_trainmatrix.shape)
		# print(tf.vocabulary_)
		# input()
		feature_names = tf.get_feature_names()
		# print(feature_names)
		tfidf_testmatrix =  tf.transform(testmatrix['Content'])
		return tfidf_trainmatrix, tfidf_testmatrix, tf

	#this extracts the features and get the vector count (frequency) for it.
	def count_vectors(self, trainmatrix, testmatrix):
		cv = CountVectorizer()
		cv_trainmatrix =  cv.fit_transform(trainmatrix['Content'])
		cv_testmatrix =  cv.transform(testmatrix['Content'])
		return cv_trainmatrix, cv_testmatrix