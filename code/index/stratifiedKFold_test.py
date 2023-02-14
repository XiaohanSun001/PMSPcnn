import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split
import joblib
np.seterr(divide='ignore',invalid='ignore')

def standardization(data,name):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x=(data - mean) / std
    np.save(name+'_mean.npy',mean)
    np.save(name+'_std.npy',std)
    #X_all normal
#    np.save(name+'_mean1.npy',mean)
#    np.save(name+'_std1.npy',std)
    return x


def load_data(workpath):

  X_D = np.load(workpath + '/X_h0.npy')
  X_D = pd.DataFrame(X_D,dtype = float)
  X_D = standardization(X_D,'X_D')
  
  X_R = np.load(workpath + '/X_h0_R.npy')
  X_R = pd.DataFrame(X_R,dtype = float)
  X_R = standardization(X_R,'X_R')

  
  Y_D = np.load(workpath+'/Y_d.npy')
  Y_D = pd.DataFrame(Y_D,dtype = float)
  
  Y_R = np.load(workpath+'/Y_R.npy')
  Y_R = pd.DataFrame(Y_R,dtype = float)
  return X_D,X_R, Y_D,Y_R


def importance(x,x_r, y,y_r,label,dim):
    #y_r=-y
    pccs=0
    n=5
    i=0
    #feature_index=
    important = np.zeros([dim,1])
    kfold = StratifiedKFold(n_splits = n, shuffle=True, random_state =9)
    for train_index, validation_index in kfold.split(x, label):
        x_train, x_validation = x.iloc[train_index], x.iloc[validation_index]
        x_R_train, x_R_validation = x_r.iloc[train_index], x_r.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
        y_R_train, y_R_validation = y_r.iloc[train_index], y_r.iloc[validation_index]
        
        X_train=np.concatenate((x_train, x_R_train),axis=0)
        X_test =np.concatenate((x_validation, x_R_validation),axis=0)
        Y_train=np.concatenate((y_train, y_R_train),axis=0)
        Y_test=np.concatenate((y_validation, y_R_validation),axis=0)
        
        
        model = xgb.XGBRegressor(objective='reg:squarederror',learning_rate=0.1, n_estimators=1600, max_depth=8, min_child_weight=4, seed=0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1)
    
       
       
        ref=model.fit(X_train, Y_train)
    
        y_pred = ref.predict(X_test)
        
        #print(y_pred.shape,Y_test.shape)
        y_pred=y_pred.flatten()
        Y_test=Y_test.flatten()
        #print(y_pred.shape,Y_test.shape)
        pccs += np.corrcoef(y_pred,Y_test)
        value=ref.feature_importances_
        important += value.reshape([dim,1])
        #print(ref.feature_importances_.shape)
        i=i+1
        feature_importance_df = pd.DataFrame()
        feature_importance_df["Feature"] = range(0,dim)
        feature_importance_df["importance"] = ref.feature_importances_
        feature_importance_df.to_csv(str(i)+'_fold_index.csv', sep=',', index=True, header=True)
     
    print("K fold average pccs: {}".format(pccs / n))
    
    #print(important.shape)
    feature_importance_df = pd.DataFrame()
    
    feature_importance_df["Feature"] = range(0,dim)
    # print(feature_importance_df["Feature"])
    # print('feature_importance_df["Feature"]')
    feature_importance_df["importance"] = important/n
    data = feature_importance_df.sort_values(by="importance", ascending=False)

    # savefile 
    
    data.to_csv('index.csv', sep=',', index=True, header=True)
    np.save('index.npy', data['Feature'])

# plot_importance(model)
  # plt.show()


def main():
 
  cutoff=12
  dim=int(cutoff*54)
  #print(dim)
  workdir='../topology_Q3211/Results'
  lable=np.load( workdir+'/label.npy')
  lable=pd.DataFrame(lable)
  
  X,X_R, Y,Y_R = load_data(workdir)
  
#  X_all=np.concatenate((X,X_R),axis=0)
#  X_all=standardization(X_all,'X_all')
  
#  X = X_all[0:3211,:]
#  X_R = X_all[3211:6422,:]
#  X=pd.DataFrame(X,dtype = float)
#  X_R=pd.DataFrame(X_R,dtype = float)
  
  print(X.shape, X_R.shape)
  importance(X,X_R,Y,Y_R,lable,dim)
  
   
if __name__ == '__main__':
    main()
