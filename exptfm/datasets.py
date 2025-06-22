from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import numpy as np 


def load_dataset_as_pandas(dataset, subset):
    dataset = load_dataset("glue", dataset)
    ds = pd.DataFrame(dataset[subset])
    scaler = MinMaxScaler()
    ds["label_ok"] = scaler.fit_transform(ds["label"].to_numpy().reshape(-1, 1))
    return ds.copy()