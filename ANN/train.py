import pandas as pd
from keras.layers import Dense, BatchNormalization, LeakyReLU, GRU, ReLU, RNN, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

file_path = '/home/harsha/Machine_Learning_Project/Default of Credit Card Clients/Dataset/defaultofcreditcardclients.csv'
df = pd.read_csv(file_path)
# df = df.iloc[1:,:]

##======= Selecting Features =======##
cor = df.corr()
cor_target = abs(cor['Y'])
relavent_features = cor_target[cor_target>0.06]


df =  df[['X1','X6','X7','X8','X9','X10','X11','X18','Y']]


##======= Normalise the Data =======##
scaler = MinMaxScaler(feature_range = (0,1))
df = scaler.fit_transform(df)

##======= Principal Component Analysis ==========##
pca = PCA(.95)
pca.fit(df[:,:-1])
inpt = pca.transform(df[:,:-1])
oupt = df[:,-1]




#Data Preparation
time_steps = 1
features = inpt.shape[1]

input_x = []
input_y = []

for i in range(0,30000,time_steps):
    sample_x = np.array(inpt[i:i+time_steps])
    sample_y = np.array(oupt[i:i+time_steps])
    input_x.append(sample_x)
    input_y.append(sample_y)

input_x = np.array(input_x)
input_y = np.array(input_y)
# input_y = np.reshape(input_y,(input_y.shape[0],input_y.shape[1]))

#Define Model Architecture
def build_model(shp):
    model = Sequential()

    # model.add(GRU(32, return_sequences = True, input_shape = (time_steps,features)))
    # model.add(ReLU())
    # model.add(BatchNormalization())

    model.add(LSTM(16, return_sequences = True, input_shape = (time_steps,features)))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(LSTM(8))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Dense(1, activation = 'sigmoid'))

    return model

##======== Build the Model =========##
model = build_model(23)
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['accuracy'])
model.summary()



##======= Splitting the Data ========##
x_train, x_val, y_train, y_val = train_test_split(input_x, input_y, test_size= 0.3, random_state= 32)

##====== Creating Checkpoints =======##
weights = '/home/harsha/Machine_Learning_Project/Default of Credit Card Clients/ANN/Weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoints = ModelCheckpoint(weights, verbose= 1, monitor= 'val_acc', save_best_only= True, mode = 'max')
callbacks = [checkpoints]

##===== Train the Model =============##
epochs = 50
batch_size = 512
model.fit(x_train,y_train, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks  = callbacks, validation_data = (x_val, y_val))
