import time
import NN
import os


class Neural:
    NNModel = None
    prediction = None
    x_train = None


def start_threaded_sniffer(treads=8):
    time.sleep(2)
    print("Unable to find threads for listening. Check connection cables\nReturning to main menu")
    return

def modelload():
    print("--------------------------------------------------------------------------\n----Main menu|Load Model|\n--------------------------------------------------------------------------")
    print("Write path of the neural network model with .json extension")
    modelpath = input()
    if(modelpath == 'back'): return None
    else:
        print("Write path of neural network weights with .h5 extension")
        weightspath = input()
        if (weightspath == 'back'): return None
        else:
            model = NN.load_model(modelpath, weightspath)
            return model

def x_train_load():
    print("--------------------------------------------------------------------------\n----Main menu|Load Input Data For Learning|\n--------------------------------------------------------------------------")
    print("Write path to the data folder. There must be only the data!")
    path = input()
    if (path == 'back'):
        return None
    else:
        try:
            os.path.exists(path)
            data = NN.set_train_data(path)
            return data
        except(Exception, OSError):
            print("Incorrect path")
            return None

def set_prediction_data():
    print("--------------------------------------------------------------------------\n----Main menu|Load Prediction Data|\n--------------------------------------------------------------------------")
    print("Write path to the prediction data folder. There must be only the data!")
    path = input()
    if (path == 'back'):
        return None
    else:
        try:
            os.path.exists(path)
            data = NN.set_prepared_prediction_dataset(path)
            return data
        except(Exception, OSError):
            print("Incorrect path")
            return None

def testnet():
    print("--------------------------------------------------------------------------\n----Main menu|Predicting Moves\n--------------------------------------------------------------------------")
    for move in Neural.prediction:
        NN.predict_move(Neural.NNModel, move)
    return

def mainmenu():
    while True:
        print("--------------------------------------------------------------------------\n----Main menu|\n--------------------------------------------------------------------------")
        command = str(input())
        if (command == 'help'):
            print("'start' - starts movement data listener\n"
                  "'loadmodel' - intiates a model loading script\n"
                  "'predictiondata' - sets the data for prediction\n"
                  "'loaddata' - sets the (x_train) data\n"
                  "'testnet' - initiates a testing of a network with model and data'\n"
                  "'netsettings' - neural network settings\n" 
                  "'exit' - exit system")
        elif (command == 'start'):
            start_threaded_sniffer()
        elif (command == 'loadmodel'):
            Neural.NNModel = modelload()
            if(Neural.NNModel):
                print("\nModel added")
        elif (command == 'loaddata'):
            Neural.x_train = x_train_load()
            try:
                if(Neural.x_train.any()): print("dataset is ready: ", Neural.x_train.shape)
            except(AttributeError): print("write the existing path")
        elif (command == "testnet"):
            if((Neural.NNModel) and (Neural.prediction)):
                testnet()
                print("Done")
            else: print("No model and(or) prediction data")

        elif (command == 'predictiondata'):
            Neural.prediction = set_prediction_data()
            if(Neural.prediction):
                print("Loaded ", len(Neural.prediction), " cases")
        elif (command == 'netsettings'):
            print("")
        elif (command == "exit"):
            break
        continue


print("SANEM v.0.0.2 Author:Plotnikov_Jaroslav, group:PO-61, SWSU")
print("Initializing system")
# time.sleep(2)
# print("Done \n")
# time.sleep(1)
print("Insert command or type 'help' to see a full list of commands")
mainmenu()