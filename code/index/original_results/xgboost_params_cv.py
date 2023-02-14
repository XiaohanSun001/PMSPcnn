#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import KFold

import random
import math
np.seterr(divide='ignore',invalid='ignore')

def standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std
#cutoff=12
def shuffle(x,y):
    x_batch=x
    y_batch=y
    seed=100
    random.seed(seed)
    random.shuffle(x_batch)
    random.seed(seed)
    random.shuffle(y_batch)
    return x_batch,y_batch

def load_data(workpath):

  X = np.load(workpath + '/X_0.npy')
  X = np.asarray(X,dtype = float)

  Y = np.load(workpath+'/Y_n.npy')
  Y = np.asarray(Y,dtype = float)
 
  
  return X, Y

def n_estimators(X, Y):
  cv_params = {'n_estimators': [1500,1600,1700,1800,1900,2000]}
  other_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
  model = xgb.XGBRegressor(**other_params)
  optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
  optimized_GBM.fit(X, Y) 
  evalute_result = optimized_GBM.cv_results_
  #print('evalute_result:{0}'.format(evalute_result))
  print('best n_estimators:{0}'.format(optimized_GBM.best_params_))
  print('best_score:{0}'.format(optimized_GBM.best_score_))
# result
#best n_estimators:{'n_estimators': 1600}
#best_score:0.737400826633247




def min_child_weight_and_max_depth(X, Y):
  cv_params = {'max_depth': [6,7,8], 'min_child_weight': [3,4,5,6,7]}
  other_params = {'learning_rate': 0.1, 'n_estimators': 1600, 'max_depth': 1, 'min_child_weight': 12, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
  model = xgb.XGBRegressor(**other_params)
  optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
  optimized_GBM.fit(X, Y)
  evalute_result = optimized_GBM.cv_results_
  #print('evalute_result:{0}'.format(evalute_result))
  print('best n_estimators:{0}'.format(optimized_GBM.best_params_))
  print('best_score:{0}'.format(optimized_GBM.best_score_))
# result
#best n_estimators:{'max_depth': 8, 'min_child_weight': 4}
#best_score:0.7507531854643907


def subsample_colsample(X,Y):
  cv_params = {'subsample': [0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'colsample_bytree': [0.2,0.3,0.4,0.5,0.6,0.7,0.8]}
  other_params = {'learning_rate': 0.1, 'n_estimators': 1600, 'max_depth': 8, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
  model = xgb.XGBRegressor(**other_params)
  optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
  optimized_GBM.fit(X, Y)
  evalute_result = optimized_GBM.cv_results_
  #print('evalute_result:{0}'.format(evalute_result))
  print('best n_estimators:{0}'.format(optimized_GBM.best_params_))
  print('best_score:{0}'.format(optimized_GBM.best_score_))
#best n_estimators:{'colsample_bytree': 0.8, 'subsample': 0.8}
#best_score:0.7507531854643907



  
def gamma(X, Y):
  cv_params = {'gamma': [0,0.05,0.1,0.2]}
  other_params = {'learning_rate': 0.1, 'n_estimators': 1600, 'max_depth': 8, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
  model = xgb.XGBRegressor(**other_params)
  optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
  optimized_GBM.fit(X, Y)
  evalute_result = optimized_GBM.cv_results_
  #print('evalute_result:{0}'.format(evalute_result))
  print('best n_estimators:{0}'.format(optimized_GBM.best_params_))
  print('best_score:{0}'.format(optimized_GBM.best_score_))

#best n_estimators:{'gamma': 0}
#best_score:0.7507531854643907




def regalpha_reglambda(X, Y):
  cv_params = {'reg_alpha': [ 0,0.1,1, 2], 'reg_lambda': [0,0.1, 1, 2]}
  other_params = {'learning_rate': 0.1, 'n_estimators': 1600, 'max_depth': 8, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
  model = xgb.XGBRegressor(**other_params)
  optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
  optimized_GBM.fit(X, Y)
  evalute_result = optimized_GBM.cv_results_
  #print('evalute_result:{0}'.format(evalute_result))
  print('best n_estimators:{0}'.format(optimized_GBM.best_params_))
  print('best_score:{0}'.format(optimized_GBM.best_score_))
#best n_estimators:{'reg_alpha': 0, 'reg_lambda': 1}
#best_score:0.7507531854643907



def learning_rate(X, Y):
  cv_params = {'learning_rate': [0.01, 0.05, 0.1, 0.2]}
  other_params = {'learning_rate': 0.1, 'n_estimators': 1600, 'max_depth': 8, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
  model = xgb.XGBRegressor(**other_params)
  optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1)
  optimized_GBM.fit(X, Y)
  evalute_result = optimized_GBM.cv_results_
  #print('evalute_result:{0}'.format(evalute_result))
  print('best n_estimators:{0}'.format(optimized_GBM.best_params_))
  print('best_score:{0}'.format(optimized_GBM.best_score_))
#best n_estimators:{'learning_rate': 0.1}
#best_score:0.7507531854643907


    

def main():
  cutoff=12
  workdir='../Results'
   
  x, Y = load_data(workdir)
  X= standardization(x)
  X, Y = shuffle(X,Y)
  print(X.shape, Y.shape)
  
  #n_estimators(X, Y)
  #  min_child_weight_and_max_depth(X, Y)
  #subsample_colsample(X,Y)
  #gamma(X, Y)
  #regalpha_reglambda(X, Y)
  learning_rate(X, Y)
  
  #Model(X, Y)




if __name__ == '__main__':
    main()


