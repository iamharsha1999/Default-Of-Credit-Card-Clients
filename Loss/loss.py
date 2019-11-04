import numpy as np

class Lossfunction:

    @staticmethod
    def softmax(x):
    	exps = np.exp(x)/np.sum(x)
    	return exps

    @staticmethod
    def cross_entropy(x, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y
        p = Loss.softmax(x)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def se(x, y):

    	"""
    	x being the output of the neural network
    	y being the actual output for the corresponding input or the label in case of classification
    	"""

    	error = (y - x)**2

    	return error

    @staticmethod
    def half_squared_error(x,y):
        """
    	x being the output of the neural network
    	y being the actual output for the corresponding input or the label in case of classification
    	"""

        error = ((y - x)**2)/2

        return error

    @staticmethod
    def absolute_error(x, y):

    	"""
    	x being the output of the neural network
    	y being the actual output for the corresponding input or the label in case of classification
    	"""
    	error = x-y
    	return abs_error
