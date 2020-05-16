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
    if (os.path.exists(datapath)):
        data_path = os.chdir(datapath)
        print(datapath)
    else:
        print("Path doesn`t exist")
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
        return x_train
    else:
        print("Path doesn`t exist")
        return None

# try: x_train = set_train_data("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements\\testmoves")
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
        return test_moves # (!)возвращает список таблиц(!)
    else:
        print("Path doesn`t exist")
# predict_moves = set_prepared_prediction_dataset("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements\\prediction")

def set_y_train(filepath):
    global y_train
    if (os.path.exists(filepath)):
        with open(filepath, "r") as file:
            y_train = []
            for line in file.readlines():
                y_train.append(int(line))
            y_train = keras.utils.to_categorical(y_train)
        print(len(y_train), y_train)
        return y_train
    else:
        print("Path doesn`t exist")
# y_train = set_y_train("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements\\prediction_output\\prediction.txt")

def set_standart_model():
    input = keras.Input(shape=(24, x_train.shape[2], 1))
    inputs = layers.Convolution2D(24, input_shape=(24,), kernel_size=(3, 1), activation="sigmoid")(input)
    hidden1 = layers.Dense(8, activation="sigmoid")(inputs)
    hidden1 = layers.Dropout(0.5)(hidden1)
    h1 = layers.Flatten()(hidden1)
    outputs = layers.Dense(6, activation="relu")(h1)
    seq_model = keras.Model(inputs=input, outputs=outputs, name="sequential_or_what")
    print(seq_model.summary())
    seq_model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return seq_model

def load_model(json_modelpath, h5weightspath=None):
    if(h5weightspath==None):
        if(os.path.exists(json_modelpath)):
            with open(json_modelpath, 'r') as json_file:
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = keras.models.model_from_json(loaded_model_json)
                print(loaded_model.summary())
                return loaded_model
        else:print("Bad path")
    else:
        if (os.path.exists(json_modelpath)):
            with open(json_modelpath, 'r') as json_file:
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = keras.models.model_from_json(loaded_model_json)
                loaded_model.load_weights(h5weightspath)
                print(loaded_model.summary())
                return loaded_model
        else:
            print("Bad path")

# model = load_model("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\model\\1model.json",
#            "C:\\Users\\NoMad\\PycharmProjects\\SANEM\\model\\1weights.h5")

def predict_move(model, move): # для move нужен массив отобранных данных
    predictArr = model.predict(move)
    # print(predictArr[0])
    maxPercentage = 0
    global movenum
    for i in enumerate(predictArr[0]):
        if (i[1] > maxPercentage):
            maxPercentage = i[1]
            movenum = i[0] + 1
    return movenum
# prednum = predict_move(model, predict_moves[0])
# print(prednum)

def fit(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

def save_json_model_h5_weights(model,modelpath=None,weightspath=None):
    if(modelpath==None & weightspath==None):
        print("Nowhere to save!")
        return
    elif(weightspath==None):
        if (os.path.exists(modelpath)):
            model_json = model.to_json()
            with open(modelpath, "w") as json_file:
                json_file.write(model_json)
            print("Model saved")
        else:
            print("Bad model path")
    elif(modelpath==None):
        if (os.path.exists(weightspath)):
            model.save_weights(weightspath)
            print("Weights saved")
        else:
            print("Bad weights path")
    else:
        if (os.path.exists(weightspath) & os.path.exists(modelpath)):
            model_json = model.to_json()
            with open(modelpath, "w") as json_file:
                json_file.write(model_json)
            model.save_weights(weightspath)
            print("Weighted model saved")
        else:
            print("Bad weights or model path")



# def define_move(predictArr):
#     maxPercentage = 0
#     global move
#     for i in enumerate(predictArr[0]):
#         if (i[1] > maxPercentage):
#             maxPercentage = i[1]
#             move = i[0] + 1
#     print(move)
#     return move
#
# def count_percents(predictArr):
#     maxPercentage = 0
#     global move
#     for i in enumerate(predictArr[0]):
#         if (i[1] > maxPercentage):
#             maxPercentage = i[1]
#             move = i[0] + 1
#     print(round(maxPercentage * 100, 2), "% this is move № ", move)