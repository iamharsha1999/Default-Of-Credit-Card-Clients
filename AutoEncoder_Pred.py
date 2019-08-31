import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from nn import AutoEncoder
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np



def mae(rowA, rowB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # errs = []
    # for i in range(len)
    err = np.sum(rowA.astype("float") - rowB.astype("float"))
    err /= float(rowA.shape[0])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


model = AutoEncoder.build(24)



weight_file = '/home/harsha/Machine_Learning_Project/Models/AutoEncoder_Weights/1_Class_Weights/weights-improvement-73-0.91.hdf5'

print("Loading Weights ....")
model.load_weights(weight_file)

data_file = '/home/harsha/Machine_Learning_Project/default of credit card clients.csv'

data = pd.read_csv(data_file)

print("Preprocessing Dataframe...")
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

data_0 = np.array(data_0.iloc[500:600, 1:25])
data_1 = np.array(data_1.iloc[500:600, 1:25])

predictions_0 = []
predictions_1 = []

predictions_0  = model.predict(data_0)

print("Original is:", data_0[0])
print("Predicted is :", predictions_0[0])
# print("Predictions on Class Label 0")
# for i in range(len(data_0)):
#     print(mae(data_0[i,:],predictions_0[i,:]))
#
predictions_1 = model.predict(data_1)

print("Original is:", data_1[0])
print("Predicted is :", predictions_1[0])
# print("Predictions on Class Label 1")
# for i in range(len(data_1)):
#     print(mae(data_1[i,:], predictions_1[i,:]))





