from pynput import keyboard as kb
import time
import numpy as np

# kb.add_hotkey('z', print([1, 1, ]), suppress=True, trigger_on_release=False)
# kb.add_hotkey('x', print([1, -1, ]), suppress=True, trigger_on_release=True)
# kb.add_hotkey('a', args=[2, 1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('s', args=[2, -1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('c', args=[3, 1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('v', args=[3, -1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('d', args=[4, 1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('f', args=[4, -1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('b', args=[5, 1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('n', args=[5, -1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('g', args=[6, 1, ], suppress=True, trigger_on_release=True)
# kb.add_hotkey('h', args=[6, -1, ], suppress=True, trigger_on_release=True)


def callb(key): #what to do on key-release
    ti1 = str(time.time() - t)[0:5] #converting float to str, slicing the float
    print("The key",key,"is pressed for",ti1,'seconds \n')
    return False #stop detecting more key-releases
def callb1(key): #what to do on key-press
    return False #stop detecting more key-presses

with kb.Listener(on_press = callb1) as listener1: #setting code for listening key-press
    listener1.join()


t = time.time() #reading time in sec

with kb.Listener(on_release = callb) as listener: #setting code for listening key-release
    listener.join()
