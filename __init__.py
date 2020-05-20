import time
import NN
import os


class Neural:
    NNModel = None
    prediction = None
    x_train = None
    y_train = None

def start_threaded_sniffer(treads=8):
    time.sleep(2)
    print("Unable to find threads for listening. Check connection cables\nReturning to main menu")
    return

def modelload():
    print("--------------------------------------------------------------------------\n----Main menu|Neural Network Settings|Load Model|\n--------------------------------------------------------------------------")
    print("Type path to the neural network model with .json extension")
    modelpath = input()
    if(modelpath == 'back'): return None
    else:
        print("Type path to the neural network weights data with .h5 extension")
        weightspath = input()
        if (weightspath == 'back'): return None
        else:
            model = NN.load_model(modelpath, weightspath)
            return model

def y_train_load():
    print("Type path to the output data file for training")
    path = input()
    if (path == 'back'): return None
    elif (os.path.exists(path)):
        try:
            Neural.y_train = NN.set_y_train(path)
            print(len(Neural.y_train), " movements loaded!")
        except(PermissionError):
            print("Write the path to the FILE, not to the folder!")
    else:
        print("Bad path")
        return None


def x_train_load():
    print("--------------------------------------------------------------------------\n----Main menu|Neural Network Settings|Load Input Data For Learning|\n--------------------------------------------------------------------------")
    print("Type path to the data folder. There must be only the data!")
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
    print("Type path to the prediction data folder. There must be only the data!")
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
    print("--------------------------------------------------------------------------\n----Main menu|Moves Prediction|\n--------------------------------------------------------------------------")
    for move in enumerate(Neural.prediction):
        num = NN.predict_move(Neural.NNModel, move[1])
        print(f"Data {move[0]+1} is a move №", num)
    return

def fit():
    print("")

def mainmenu():
    while True:
        print("--------------------------------------------------------------------------\n----Main menu|\n--------------------------------------------------------------------------")
        command = str(input())
        if (command == 'help'):
            print("'start' - starts movement data listener\n"
                  "'predictiondata' - sets the data for prediction\n"
                  "'testnet' - initiates a testing of a network with model and data'\n"
                  "'netset' - neural network settings\n" 
                  "'exit' - exit system\n")
        elif (command == 'start'):
            start_threaded_sniffer()
        elif (command == "testnet"):
            if((Neural.NNModel) and (Neural.prediction)):
                testnet()
                print("Done")
            else: print("No model and(or) prediction data")

        elif (command == 'predictiondata'):
            Neural.prediction = set_prediction_data()
            if(Neural.prediction):
                print("Loaded ", len(Neural.prediction), " cases")
        elif (command == 'netset'):
            while True:
                print("--------------------------------------------------------------------------\n----Main menu|Neural Network Settings|\n--------------------------------------------------------------------------")
                command = input()
                if(command == 'loaddata'):
                    Neural.x_train = x_train_load()
                    try:
                        if (Neural.x_train.any()): print("dataset is ready: ", Neural.x_train.shape)
                    except(AttributeError):
                        print("write the existing path")
                elif (command == 'loadmodel'):
                    Neural.NNModel = modelload()
                    if (Neural.NNModel):
                        print("\nModel added")
                elif (command == 'loadoutdata'):
                    y_train_load()
                elif (command == 'help'):
                    print("'loadmodel' - intiates a model loading script\n"
                          "'loaddata' - sets the input data for training\n"
                          "'loadoutdata' - sets the output data for training\n"
                          "'back' - back to main menu\n")
                elif (command == 'back'):
                    break
                continue
            print("") # сюда добавить обучение сети и добавить выбор файла для prediction_output
        elif(command == "exit"):
            break
        continue


print("SANEM v.0.0.2 Author:Plotnikov_Jaroslav, group:PO-61, SWSU")
print("Initializing system")
# time.sleep(2)
# print("Done \n")
# time.sleep(1)
print("Insert command or type 'help' to see a full list of commands")
mainmenu()