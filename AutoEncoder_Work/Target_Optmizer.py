from nn import AutoEncoder
import numpy as np
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split

opt = AutoEncoder.target_optimizer(1)
model = AutoEncoder.build(24)


opt_path = '/home/harsha/Machine_Learning_Project/Models/Target_Optimizer/27th_August/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
ae_model_path = '/home/harsha/Machine_Learning_Project/Models/AutoEncoder_Weights/6th_August_MAE/weights-improvement-82-0.98.hdf5'

##============ (Loading The Weights============#
model.load_weights(ae_model_path)


##=============Checkpoint Creation==============##


checkpoint = ModelCheckpoint(opt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

##=========AutoEncoder Prdiction================##
data_file = '/home/harsha/Machine_Learning_Project/default of credit card clients.csv'

data = pd.read_csv(data_file)
print("Preprocessing Dataframe...")
data.drop(data.columns[0],axis= 1,inplace=True)
data_0 = data.where(data['Y'] == '0')
data_1 = data.where(data['Y'] == '1')
data_0.dropna(inplace=True)
data_1.dropna(inplace=True)



convert_dict = {'X1':float, 'X2':float, 'X3':float, 'X4':float,  'X5':float, 'X6':float, 'X7':float, 'X8':float, 'X9':float, 'X10':float,
                'X11':float, 'X12':float, 'X13':float, 'X14':float, 'X15':float, 'X16':float, 'X17':float, 'X18':float, 'X19':float, 'X20':float,
                'X21':float, 'X22':float, 'X23':float, 'Y':float
                }

data_0 = data_0.astype(convert_dict)
data_1 = data_1.astype(convert_dict)

data_0_opt = np.array(data_0.iloc[:, 24])
data_1_opt = np.array(data_1.iloc[:, 24])
data_0 = np.array(data_0.iloc[:, 1:25])
data_1 = np.array(data_1.iloc[:, 1:25])

predictions_0 = []
predictions_1 = []

predictions_0 = model.predict(data_0)
predictions_1 = model.predict(data_1)



##==================Split the Data==============##
train_X_0, valid_X_0, train_ground_0, valid_ground_0 = train_test_split(data_0_opt, predictions_0[:, 23].T, test_size=0.2, random_state=13)
train_X_1,valid_X_1,train_ground_1,valid_ground_1 = train_test_split(data_1_opt, predictions_1[:, 23].T, test_size=0.2, random_state=13)
print("Data Splitted")
#===============Training the Model==============##

print("Training Model...")
opt.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

batch_size = 25
epochs = 100

opt_train_0 = opt.fit(train_X_0,train_ground_0, batch_size=batch_size, epochs= epochs, callbacks=callbacks_list, verbose=1, validation_data=(valid_X_0, valid_ground_0))
# opt_train_0 = opt.fit(predictions_0[23], train_X_0['Y'], batch_size=batch_size, epochs= epochs, callbacks=callbacks_list, verbose=1, validation_data=(valid_X_1, valid_ground_1))
