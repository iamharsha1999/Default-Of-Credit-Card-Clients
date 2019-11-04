import pandas as pd
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = '/home/harsha/Machine_Learning_Project/Default of Credit Card Clients/Dataset/defaultofcreditcardclients.csv'
df = pd.read_csv(file_path)
df = df.iloc[1:,:]

##Define Model Architecture
def build_model(shp):
    model = Sequential()

    model.add(Dense(32, activation = 'relu', input_dim = shp))
    model.add(BatchNormalization())

    model.add(Dense(16, activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Dense(8, activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation = 'sigmoid'))

    return model

##======== Build the Model =========##
model = build_model(23)
model.compile(loss = 'binary_crossentropy', optimizer= 'Nadam', metrics = ['accuracy'])
model.summary()

##======= Normalise the Data =======##
scaler = MinMaxScaler(feature_range = (0,1))
df = scaler.fit_transform(df)

##======= Splitting the Data ========##
x_train, x_val, y_train, y_val = train_test_split(df[:,0:23], df[:,-1], test_size= 0.3, random_state= 32)

##====== Creating Checkpoints =======##
weights = '/home/harsha/Machine_Learning_Project/Default of Credit Card Clients/ANN/Weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoints = ModelCheckpoint(weights, verbose= 1, monitor= 'val_acc', save_best_only= True, mode = 'max')
callbacks = [checkpoints]

##===== Train the Model =============##
epochs = 50
batch_size = 512
model.fit(x_train,y_train, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks  = callbacks, validation_data = (x_val, y_val))
