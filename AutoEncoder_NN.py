import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from nn import AutoEncoder
import numpy as np



## ================= Data Preprocessing ==================##

data = pd.read_csv('/home/harsha/Machine_Learning_Project/Dataset/default of credit card clients.csv')

# data.drop(data.columns[0],axis= 1,inplace=True)
data_0 = data.where(data['Y'] == '0')
# data_1 = data.where(data['Y'] == '1')
data_0.dropna(inplace=True)
data.dropna(inplace=True)

data_0 = data_0.iloc[1:, :]
data = data.iloc[1:, :]

convert_dict = {'X1':float, 'X2':float, 'X3':float, 'X4':float,  'X5':float, 'X6':float, 'X7':float, 'X8':float, 'X9':float, 'X10':float,
                'X11':float, 'X12':float, 'X13':float, 'X14':float, 'X15':float, 'X16':float, 'X17':float, 'X18':float, 'X19':float, 'X20':float, 'X21':float, 'X22':float, 'X23':float, 'Y':float
                }
data_0 = data_0.astype(convert_dict)
data = data.astype(convert_dict)



## ====================Create the model ==================##

model = AutoEncoder.build(24)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

filepath = '/home/harsha/Machine_Learning_Project/Models/AutoEncoder_Weights/Mix_Data_Weights/31st_August/Batch_Size_100/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

## ==================== Split the Dataset ================##

# train_X_0,valid_X_0,train_ground_0,valid_ground_0 = train_test_split(data_0, data_0, test_size=0.2, random_state=13)
train_X,valid_X,train_ground,valid_ground = train_test_split(data, data, test_size=0.2, random_state=13)

# np_file_path = '/home/harsha/Machine_Learning_Project/Numpy_Files'

# np.load(np_file_path + '/train_X_0.npy')
# train_X_1 = np.load(np_file_path + '/train_X_1.npy')
# np.load(np_file_path + '/valid_X_0.npy')
# valid_X_1 = np.load(np_file_path + '/valid_X_1.npy')
# np.load(np_file_path + '/train_ground_0.npy')
# train_ground_1  = np.load(np_file_path + '/train_ground_1.npy')
# np.load(np_file_path + '/valid_ground_0.npy')
# valid_ground_1 = np.load(np_file_path + '/valid_ground_1.npy')
# print("Loaded Files")

## ==================== Train the model ===================##
batch_size = 100
epochs = 100


# model_train_0 = model.fit(train_X_0, train_ground_0, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(valid_X_0, valid_ground_0))
model_train_1 = model.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(valid_X, valid_ground))






