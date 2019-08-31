from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy

def nn_model():
    model = Sequential()

    model.add(Dense(13, input_dim=(13,)))

    model.add(Dense(8, activation='relu'))

    model.add(Dense(4, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model

csv_file_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/heart.csv'
df = pd.read_csv(csv_file_path)
corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

plt.show()




