import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


def read_data(id):
    binary_file_path = "binary\\"
    multiclass_file_path = "multiclass\\"

    if id == 1:
        file_path = binary_file_path
        key = {0: 'book', 1: 'plastic case'}
    elif id == 2:
        file_path = multiclass_file_path
        key = {0: 'air', 1: 'book', 2: 'hand', 3: 'knife', 4: 'plastic case'}

    X = pd.read_csv(file_path+"X.csv", header=None)
    XToClassify = pd.read_csv(file_path+"XToClassify.csv", header=None)
    y = pd.read_csv(file_path+"y.csv", header=None)
    

if __name__ == "__main__":
    read_data(1)
