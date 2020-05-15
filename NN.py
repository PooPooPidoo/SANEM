import numpy as np
import os
import pandas as pd
import dotenv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
dotenv.load_dotenv("vars.env")
varpath = dotenv.find_dotenv("vars.env")

def set_datapath(datapath):
    data_path = os.chdir(datapath)
    print(datapath)
#set_datapath("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements")

def set_train_data(datapath):
    if(os.path.exists(datapath)):
        os.chdir(datapath)
        path = os.listdir(datapath)
        global x_train, data1, maxlen
        maxlen = 0
        flag = False

        for file in path:
            data1 = np.array(pd.read_csv(file, sep=','))
            data1 = np.array([data1[:, 42:].transpose()])
            if (maxlen < data1.shape[2]):
                maxlen = data1.shape[2]
        dotenv.set_key(varpath,"LEN",str(maxlen),"always")
        for file in path:
            if (flag == False):
                x_train = np.zeros((data1.shape[0], data1.shape[1], maxlen))
                x_train[:data1.shape[0], :data1.shape[1], :data1.shape[2]] = data1
                flag = True
                continue
            data1 = np.array(pd.read_csv(file, sep=','))
            data1 = np.array([data1[:, 42:].transpose()])
            if (data1.shape[2] < maxlen):
                newdata = np.zeros((data1.shape[0], data1.shape[1], maxlen))
                newdata[:data1.shape[0], :data1.shape[1], :data1.shape[2]] = data1
                x_train = np.append(x_train, newdata, axis=0)
            elif (data1.shape[2] == maxlen):
                x_train = np.append(x_train, data1, axis=0)

        print("massive loaded: ",x_train.shape,"\nPreparing dataset")
        x_train = x_train.reshape((x_train.shape[0], 24, x_train.shape[2], 1))
        x_train = x_train.astype('float32')
        print("dataset is ready: ",x_train.shape)
        return x_train
    else: print("Path doesn`t exist")

# try:set_train_data("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements\\testmoves")
# except(UnicodeError, SyntaxError): print("Incorrect path")

def set_prepared_prediction_dataset(datapath):
    if (os.path.exists(datapath)):
        os.chdir(datapath)
        path = os.listdir(datapath)
        maxlen = int(dotenv.get_key(varpath,"LEN"))
        test_moves = list()
        for file in enumerate(path):
            predict_move = np.array(pd.read_csv(file[1], sep=','))
            predict_move = np.array([predict_move[:maxlen, 42:].transpose()])
            if (predict_move.shape[2] < maxlen):
                newdata = np.zeros((predict_move.shape[0], predict_move.shape[1], maxlen))
                newdata[:predict_move.shape[0], :predict_move.shape[1], :predict_move.shape[2]] = predict_move
                predict_move = newdata
            predict_move = predict_move.reshape((predict_move.shape[0], 24, predict_move.shape[2], 1))
            predict_move = predict_move.astype('float32')
            test_moves.append(predict_move)
            print(f"move {file[0]+1} added!")
        print(len(test_moves))
        return test_moves
    else:
        print("Path doesn`t exist")
# try:set_prepared_prediction_dataset("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements\\prediction")
# except(UnicodeError, SyntaxError): print("Incorrect path")
