import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv("vars.env")
DATAPATH = os.getenv("DATAPATH")

dataname = 'm01_s01_e01_positions.txt'
separator = ','

def load_data (datapath, dataname, separator):
    data_path = os.chdir(datapath)
    table = np.array(pd.read_csv(dataname, sep=separator))
    table = np.array(table[:, 42:])
    return table


def tablesplit (table, quantity):
    cols = np.hsplit(table, indices_or_sections=quantity)
    return cols


def formAndSaveFigure (threeDimsTable, name, datapath=None):
    if (datapath != None): os.chdir(datapath)
    fig = mpl.pyplot.figure()
    ax = fig.gca(projection='3d')
    x = threeDimsTable[0]
    y = threeDimsTable[1]
    z = threeDimsTable[2]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    mpl.pyplot.show()
    fig.savefig(name)

triplecols = tablesplit(load_data(DATAPATH, dataname, separator),8) # деление массива на 8 массивов по 3 столбца в каждом

for n in enumerate(triplecols):
    print(n[0])
    formAndSaveFigure(triplecols[n[0]].transpose(), f'sensor{n[0]}.png')


