import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_path = os.chdir("C:\\Users\\NoMad\\PycharmProjects\\SANEM\\movements")

# сборка входного набора из 25 таблиц(24 столбца, n строк) с движениями (первые 5 - первое движение, дальнейших движений - по 4), лучше не придумал
# (25, n, 24)
x_train = np.array(pd.read_csv('m01_s01_e01_positions.txt', sep=','))
x_train = np.array([x_train[:, 42:]])
print(x_train.shape)

for i in range(1,7):
    for j in range(2,6):
        data1 = np.array(pd.read_csv(f"m0{i}_s0{j}_e01_positions.txt", sep=','))
        data1 = np.array([data1[:, 42:]])
        if(x_train.shape[1]>data1.shape[1]):
            x_train = x_train[:,:data1.shape[1],:]
            x_train = np.append(x_train, data1, axis=0)
        elif(x_train.shape[1]<data1.shape[1]):
            data1 = data1[:,:x_train.shape[1],:]
            x_train = np.append(x_train, data1, axis=0)

print(x_train.shape)

# другой вариант таблицы, где 25 - число движений, 24 - число пространственных переменных, n - число моментов, потребовавшихся для записи движения
# (25, 24, n)
x_train = np.array(pd.read_csv('m01_s01_e01_positions.txt', sep=','))
x_train = np.array([x_train[:, 42:].transpose()])
print(x_train.shape)

for i in range(1,7):
    for j in range(2,6):
        data1 = np.array(pd.read_csv(f"m0{i}_s0{j}_e01_positions.txt", sep=','))
        data1 = np.array([data1[:, 42:].transpose()])

        if(x_train.shape[2]>data1.shape[2]):
            x_train = x_train[:,:,:data1.shape[2]]
            x_train = np.append(x_train, data1, axis=0)
        elif(x_train.shape[2]<data1.shape[2]):
            data1 = data1[:,:,:x_train.shape[2]]
            x_train = np.append(x_train, data1, axis=0)

print(x_train.shape)

# выходной набор данных. Столбец с единицей соответствует номеру соответствующего движения (движений всего 6)
y_train = [[1,0,0,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0],
           [0,1,0,0,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0],
           [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0],
           [0,0,0,1,0,0], [0,0,0,1,0,0], [0,0,0,1,0,0], [0,0,0,1,0,0],
           [0,0,0,0,1,0], [0,0,0,0,1,0], [0,0,0,0,1,0], [0,0,0,0,1,0],
           [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1]]
y_train = np.array(y_train, dtype=float)
print(y_train)


# получение нужной таблицы данных в формате numpy_array
data = np.array(pd.read_csv('m01_s01_e01_positions.txt', sep=','))
data = np.array(data[:,42:])
print(data)
print(np.shape(data))

# построение модели сети
input_layer = layers.Dense(25, input_shape=(25,24,), dtype=float) #входной слой, таблица из 24 колонок
hidden_layer1 = layers.Dense(shape=(25,24), activation="relu") # скрытый
h1 = hidden_layer1(input_layer)                     # слой, 24 нейрона
h1 = layers.Dense(8, activation="sigmoid")(h1)      # следующий скрытый слой, 8 нейронов
output_layer = layers.Dense(6)(h1)                  # выход, предполагается в формате [0,0,0,0,1,0], где 1 - распознанное движение
model = keras.Model(inputs=input_layer, outputs=output_layer, name="first_model")
model.summary()


keras.utils.plot_model(model, "first_model.png")
keras.utils.plot_model(model, "first_model_with_shape_info.png", show_shapes=True)


# сбор модели
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# другой вариант нейросети
input = keras.Input(shape=(24,))
inputs = layers.Convolution2D(25, input_shape=(24,), kernel_size=(3,1), activation="relu")(input)
hidden1 = layers.Convolution2D(25, kernel_size=(3,1), activation="sigmoid")
h1 = hidden1(inputs)
h1 = layers.Flatten()(h1)
outputs = layers.Dense(6, activation="sigmoid")

seq_model = keras.Model(inputs=input, outputs=outputs, name="sequential_or_what")

model.fit(x_train,y_train,epochs=100,batch_size=25)
