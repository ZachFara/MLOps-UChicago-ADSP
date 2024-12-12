import kagglehub
from pathlib import Path
import shutil
import os
import glob
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

def main():
    h2o.init(port = 54326, nthreads = -1, max_mem_size = "6g")
    train = h2o.import_file("data/train.csv")
    test = h2o.import_file("data/test.csv")
    x = train.columns
    y = "Productivity Lost"
    if y in x:
        x.remove(y)
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()
    aml = H2OAutoML(max_models=100, max_runtime_secs=60, seed=1) 
    aml.train(x=x, y=y, training_frame=train)
    lb = aml.leaderboard
    print(lb.head(rows=25))

if __name__ == '__main__':
    main()