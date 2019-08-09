import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_absolute_error
from sklearn.model_selection import train_test_split

def autoencoder():
    model = Sequential()

    model.add(Dense(24, activation='relu', input_dim=24))

    model.add(Dense(12, activation='relu'))

    # model.add(Dense(6, activation='relu'))
    #
    # model.add(Dense(3,activation='relu'))
    #
    # model.add(Dense(6, activation='relu'))

    model.add(Dense(12, activation='relu'))

    model.add(Dense(24, activation='relu'))

    model.compile(loss='mean_absolute_error', optimizer = 'adam' , metrics=['accuracy'])

    return model


## ================= Data Preprocessing ==================##

data = pd.read_csv('/home/harsha/Machine_Learning_Project/default of credit card clients.csv')

data.drop(data.columns[0],axis= 1,inplace=True)
data_0 = data.where(data['Y'] == '0')
data_1 = data.where(data['Y'] == '1')
data_0.dropna(inplace=True)
data_1.dropna(inplace=True)



convert_dict = {'X1':float, 'X2':float, 'X3':float, 'X4':float,  'X5':float, 'X6':float, 'X7':float, 'X8':float, 'X9':float, 'X10':float,
                'X11':float, 'X12':float, 'X13':float, 'X14':float, 'X15':float, 'X16':float, 'X17':float, 'X18':float, 'X19':float, 'X20':float, 'X21':float, 'X22':float, 'X23':float, 'Y':float
                }
data_0 = data_0.astype(convert_dict)
data_1 = data_1.astype(convert_dict)

data_0 = data_0.iloc[:, 1:25]
data_1 = data_1.iloc[:, 1:25]

## ====================Create the model ==================##

model = autoencoder()

filepath = '/home/harsha/Machine_Learning_Project/Models/6th_August_MAE/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

## ==================== Split the Dataset ================##

train_X_0,valid_X_0,train_ground_0,valid_ground_0 = train_test_split(data_0, data_0, test_size=0.2, random_state=13)
train_X_1,valid_X_1,train_ground_1,valid_ground_1 = train_test_split(data_0, data_0, test_size=0.2, random_state=13)

## ==================== Train the model ===================##
batch_size = 25
epochs = 100


model_train_0 = model.fit(train_X_0, train_ground_0, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(valid_X_0, valid_ground_0))
# model_train_1 = model.fit(train_X_1, train_ground_1, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(valid_X_1, valid_ground_1))



