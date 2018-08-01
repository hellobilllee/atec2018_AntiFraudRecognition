import pandas as pd
import numpy  as np

if __name__=="__main__":
    # dtype = {"id":np.str,"label":np.uint8}
    # f_16 =  [23,43,44,45,56,57,58,66,67,68,69,70,71,72,73,74,75,78,79,80,81,
    #           102,103,104,105,106,107,108,109,110,154,162,163,164,
    #           205,206,207,208,209,210,212,213,214,215,216,217,218,227,228,229,231,232,233,234]+[e for e in range(235,298)]
    # for i in f_16:
    #     dtype.update({"f" + str(i): np.uint16})
    # f_32 = [5,82,83,84,85,86]
    # for i in f_32:
    #     dtype.update({"f" + str(i): np.float32})
    # f_8 = f_16+f_32
    # for i in [e for e in range(1,298) if e not in f_8]:
    #     dtype.update({"f"+str(i):np.uint8})
    # df = pd.read_csv("../sourcedata/atec_anti_fraud_train.csv",header=0,index_col=None)
    df = pd.read_csv("../sourcedata/atec_anti_fraud_test_b.csv",header=0,index_col=None)
    ds = df.describe()
    print(ds)
    ds.to_csv("../statsfile/stats1.csv",mode="a")
    print()
