import pandas as pd
from sklearn import preprocessing

from abc import ABC, abstractmethod


def build_y_X(df, cat_vars:list, target_var:str, from_test:str):
    df_u = df.copy()
    # Dummies
    df_u[cat_vars] = df_u[cat_vars].astype('object')
    df_dummies = pd.get_dummies(df_u[cat_vars])
    
    # Numeric vars
    df_numeric = df_u.drop(['Wilderness_Area', 'Soil_Type'], axis=1)
    
    # Split numeric into test and training
    df_numeric_test = df_numeric[df_numeric[from_test] == 'test'].copy()
    df_numeric_train = df_numeric[df_numeric[from_test] == 'train'].copy()

    # Delete from
    df_numeric_train.drop(from_test, axis=1, inplace=True)
    df_numeric_test.drop(from_test, axis=1, inplace=True)
    
    # Delete the target var
    target = df_numeric_train[target_var]
    df_numeric_train.drop(target_var, axis=1, inplace=True)
    df_numeric_test.drop(target_var, axis=1, inplace=True)
    
    # Scaling the features

    scaler = preprocessing.StandardScaler()
    scaler.fit(df_numeric_train)

    df_numeric_test_s = pd.DataFrame(scaler.transform(df_numeric_test), 
                                    index = df_numeric_test.index,
                                    columns = df_numeric_test.columns)
    df_numeric_train_s = pd.DataFrame(scaler.transform(df_numeric_train), 
                                      index = df_numeric_train.index,
                                      columns = df_numeric_train.columns)

    # Concat numeric and dummies
    df_test_f = pd.merge(df_numeric_test_s, df_dummies, left_index = True, right_index = True)
    df_train_f = pd.merge(df_numeric_train_s, df_dummies, left_index = True, right_index = True)
    
    return target, df_train_f, df_test_f

class Transform(ABC):
   
    @abstractmethod
    def transform(self):
        return NotImplementedError

class ReScaling(Transform):
    def __init__(self, kind: str):
        self.kind = kind
    
    def transform(self, df):
        self.df = df
        if self.kind == 'min_max':
            for feature in (self.df.columns):
                self.df[feature] = (self.df[feature] - self.df[feature].min()) / (self.df[feature].max() - self.df[feature].min())
            return self.df
        elif self.kind == 'standarization':
            for feature in (self.df.columns):
                self.df[feature] = (self.df[feature] - self.df[feature].mean()) / self.df[feature].std()
            return self.df


class PolyFeatures(Transform):
    def __init__(self, n_poly_f: int):
        self.n_poly_f = n_poly_f
        
    def transform(self, df, columns:list):
        self.df = df
        self.columns = columns
        for feature in self.columns:
            for i_poly in range(2,self.n_poly_f+1):
                self.df[feature+'_poly'+str(i_poly)] = self.df[feature] ** i_poly
        return self.df


def reweight_proba(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w

def saving(name, model, X_test, X_train, y_train, y_hat_test):
    joblib.dump(model, 'models/'+name+'.sav')

    # Saving the data that the model uses
    X_train.to_csv('data/output/'+name+'_X_train.csv', index=False)
    y_train.to_csv('data/output/'+name+'_y_train.csv', index=False)
    X_test.to_csv('data/output/'+name+'_X_test.csv', index=False)

    # Step 9: Produce .csv for kaggle testing 
    test_predictions_submit = pd.DataFrame({"Index": X_test.index, "Cover_Type": y_hat_test})
    test_predictions_submit.to_csv('data/output/'+name+'_y_hat_test.csv', index = False)