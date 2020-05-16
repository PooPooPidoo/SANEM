import time
import NN
import os

def start_threaded_sniffer(treads=8):
    print("Unable to find threads for listening. Check connection cables\nReturning to main menu")
    time.sleep(1)
    mainmenu()

def modelload():
    print("--------------------------------------------------------------------------\n----Main menu|LoadModel|\n--------------------------------------------------------------------------")
    print("Write path of the neural network model with .json extension")
    modelpath = input()
    if(modelpath == 'back'): return None
    else:
        print("Write path of neural network weights with .h5 extension")
        weightspath = input()
        if (weightspath == 'back'): return None
        else:
            NNMOdel = NN.load_model(modelpath, weightspath)
            return NNMOdel

def x_train_load():
    print("--------------------------------------------------------------------------\n----Main menu|LoadData|\n--------------------------------------------------------------------------")
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


def mainmenu():
    while True:
        print("--------------------------------------------------------------------------\n----Main menu|\n--------------------------------------------------------------------------")
        command = str(input())
        if (command == 'help'):
            print("'start' - starts movement data listener\n"
                  "'loadmodel' - intiates a model loading script\n"
                  "'loaddata' - initiates data loading\n"
                  "'testnet' - initiates a testing of a network with model and data'\n"
                  "'netsettings' - neural network settings\n"
                  "'exit' - exit system")
        elif (command == 'start'):
            start_threaded_sniffer()
        elif (command == 'loadmodel'):
            NNModel = modelload()
            if(NNModel):
                print("\nModel added")
        elif (command == 'loaddata'):
            x_train = x_train_load()
            try:
                if(x_train.any()): print("dataset is ready: ", x_train.shape)
            except(AttributeError): print("write the existing path")
        elif (command == "testnet"):
            print("")
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