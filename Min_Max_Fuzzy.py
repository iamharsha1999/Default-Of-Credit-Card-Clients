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
df_path = 'Dataset/defaultofcreditcardclients.csv'
df = pd.read_csv(df_path, header=1)
y = df.iloc[:,-1]
x = df.iloc[:,:-1]
x = fuzzfy(x)


#Initaing the Model
fuzzy_model = MinMaxFuzzy(x,y)

#Fitting the Data to train the model
fuzzy_model.fit(x, y, 'AND',100, optmizer = 'sgd')
