import time

def start_threaded_sniffer(treads=8):
    print("Unable to find ")

print("SANEM v.0.0.2 Author:Plotnikov_Jaroslav, group:PO-61, SWSU")
print("Initializing system")
time.sleep(1)
print("Done \nInsert command or type 'help' to see a full list of commands")
command = input()
if(command == 'help'):
    print("'start' - starts movement data listener\n"
        "'loadmodel' - intiates a model loading script\n"
        "'loaddata' - initiates data loading\n"
        "'testnet' - initiates a testing of a network with model and data'")
elif(command == 'start'):
    print("Unable to find threads for listening. Check connection cables")
elif(command == 'loadmodel'):
    print("")
elif(command == 'loaddata'):
    print("")
elif(command == 'testnet'):
    print("")


command = input()