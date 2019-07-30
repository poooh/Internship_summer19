from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Base class for linear classifiers
class Classifiers(object):

	def naivebayes(self):
		clf = GaussianNB()
		return clf

	def linearclassifier(self):
		clf = linear_model.SGDClassifier()
		return clf

	def ridgeclassifier(self):
		clf = linear_model.RidgeClassifier()
		return clf

	def linearsvc(self):
		clf = LinearSVC()
		return clf

	def logisticregression(self):
		clf = LogisticRegression()
		return clf

	def svm(self):
		clf = SVC(kernel='linear',  probability=True)
		return clf

	def randomforest(self):
		clf = RandomForestClassifier()
		return clf