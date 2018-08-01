import gc
# import fancyimpute as fi
import numpy as np
import pandas as pd
# from knnimpute import knn_impute_few_observed
from ycimpute.imputer import knnimput
from fancyimpute import MICE
def select_feature():
    complete_cols = []
    for col in ["f" + str(j) for j in range(1, 20) if j != 5]:
        complete_cols.append(col)
    df_train = pd.read_csv("../transdata/train_impute_v1.csv",header=0,index_col=None)
    print(df_train.describe())
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    # df_train =fi.KNN(k=2).complete(df_train)
    # df_train = knn_impute_few_observed(df_train,k=3,missing_mask=df_train.shape)

    train_cols = list(set(set(df_train.columns)-set("label"))-set(complete_cols))
    df_cols = df_train[complete_cols]
    for col in train_cols:
        print(col)
        impute_col = complete_cols
        impute_col.append(col)
        df_col = df_train[impute_col]
        da_col = MICE().complete(df_col.values)
        df_cols[col] = pd.Series(da_col[:,-1])
    # df_train = knnimput.KNN(k=1).complete(df_train.values)
    # df_train = pd.DataFrame(df_train,columns=train_cols)
    df_train.to_csv("../transdata/train_imputed_v2.csv",header=True,index=False)
    print("imputation over!")
    del df_train
    gc.collect()
    df_test = pd.read_csv("../transdata/test_impute_v1.csv",header=0,index_col=None).astype(np.float16)
    print(df_test.describe())
    # df_test.drop(labels=["date"], axis=1, inplace=True)
    # df_test = fi.KNN(k=3).complete(df_test)
    df_test.to_csv("../transdata/test_imputed_v2.csv", header=True, index=False)
    return
if __name__ == "__main__":
    select_feature()