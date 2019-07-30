import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# # specify parameters and distributions to sample from
# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, 11),
#               "min_samples_split": sp_randint(2, 11),
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}

# # run randomized search
# n_iter_search = 20
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=5, iid=False)

# start = time()
# random_search.fit(X, y)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
# report(random_search.cv_results_)

# use a full grid over all parameters
def gridsearch(tfidf_trainmatrix,trainmatrix, clf):
  # param_grid = {"loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
  #               "penalty": ["l2", "l1","elasticnet"],
  #               "alpha": [0.0001, 0.001, 0.01,0.1,0.00001],
  #               "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
  #               "max_iter": [100,1000, 10000],
  #               "eta0" : [0.0001, 0.001, 0.01,0.1]}

  # param_grid = {"alpha": [0.0001, 0.001, 0.01,0.1,0.00001],
  #               "tol": [0.0001, 0.001, 0.01,0.1,0.00001],
  #               "power_t": [0.4, 0.5, 0.6, 0.7],
  #               "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
  #               "eta0" : [0.0001, 0.001, 0.01,0.1, 1, 10]}

  # param_grid = {"tol": [0.0001, 0.001, 0.01,0.1,0.00001],
  #             "C": [1, 3, 5, 10, 100, 1000],
  #             "intercept_scaling": ["constant", "optimal", "invscaling", "adaptive"],
  #             "random_state" : [0.0001, 0.001, 0.01,0.1, 1, 10]}

  param_grid = {"tol": [0.0001, 0.001, 0.01,0.1,0.00001],
              "C": [1, 3, 5, 10, 100, 1000]}

  # param_grid = {"eta0" : [0.0001, 0.001, 0.01,0.1, 1, 10]}

  # run grid search
  # grid_search = GridSearchCV(clf, param_grid=param_grid, scoring = "accuracy",verbose=100, cv=5, iid=False)
  grid_search = GridSearchCV(clf, param_grid=param_grid, scoring = "accuracy",verbose=100, iid=False)
  start = time()
  grid_search.fit(tfidf_trainmatrix, trainmatrix['Classname'])

  print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (time() - start, len(grid_search.cv_results_['params'])))
  report(grid_search.cv_results_)