from keras.models import Sequential
from keras.layers.core import Dense

class AutoEncoder:
    @staticmethod
    def build(input_dim):
        model = Sequential()

        model.add(Dense(24, activation='relu', input_dim=input_dim))

        model.add(Dense(12, activation='relu'))

        # model.add(Dense(6, activation='relu'))
        #
        # model.add(Dense(3,activation='relu'))
        #
        # model.add(Dense(6, activation='relu'))

        model.add(Dense(12, activation='relu'))

        model.add(Dense(24, activation='relu'))

        return model

    @staticmethod
    def target_optimizer(input_dim):

        model = Sequential()

        model.add(Dense(5, activation='relu', input_dim=input_dim))

        model.add(Dense(1, activation='relu'))

        return model
