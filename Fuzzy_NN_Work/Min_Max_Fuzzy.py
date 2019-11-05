import numpy as np
import pandas as pd
from Fuzzy import MinMaxFuzzy

def fuzzfy(df):
    columns = list(df)

    for col in columns:
        maximum = df[col].max()
        minimum = df[col].min()

        df[col] = np.around((df[col] - minimum)/(maximum - minimum) , decimals = 4)

    return df

## Reading the Data
df_path = '/home/harsha/Machine_Learning_Project/AI_J_Component/cardio_train.csvv'
df = pd.read_csv(df_path)
print(df)
y = df.iloc[:,-1]
x = df.iloc[:,:-1]

x = fuzzfy(x)
# print("Main",x.shape)
# print(y.shape)


#Initaing the Model
fuzzy_model = MinMaxFuzzy(x,y)

#Fitting the Data to train the model
fuzzy_model.fit(x, y, 'AND',100, optimizer = 'sgd')
