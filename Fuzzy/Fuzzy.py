import numpy as np 

class MinMaxFuzzy:
	
	
	def __init__(self, x, y):

		self.input = x
		self.weights1 = np.random.randn(self.input.shape[1], 32) ##Weights for connections from Input to 1st hidden layer
		self.weights2 = np.random.randn(32, 12)  ##Weights for connections from 1st Hidden Layer to 2nd Hidden Layer
		self.weights3 = np.random.randn(12, 1)  ##Weights for connections from 2nd Hidden Layer to 3rd Hidden Layer
		self.y = y
		self.output = np.zeroes(self.y.shape)
	
	def cpneuron(output_previous, b, neuronfunction):

		cmps = []
		
		if neuronfunction == 'OR':

			for j in range(b.shape[1]):
				ncmps = 0
				for i in range(output_previous.shape[1]):
					ncmps = max(ncmps, min(b[j][i], output_previous[i]))
				cmps.append(max(ncmps))
		
		elif neuronfunction == 'AND':

			for j in range(b.shape[1]):
				ncmps = 1
				for i in range(output_previous.shape[1]):
					ncmps = min(ncmps, max(b[j], output_previous[i]))
				cmps.append(min(ncmps))

		cmps = np.array(cmps)

	

		return cmps


	def feedforward(self, activation):

		if activation = "OR":

			self.layer1 = cpneuron(self.input, self.weights1, 'OR')
			self.layer2 = cpneuron(self.layer1, self.weights2, 'OR')
			self.output = cpneuron(self.layer2, self.weights3, 'OR')

		elif activation = "AND":
			
			self.layer1 = cpneuron(self.input, self.weights1, 'AND')
			self.layer2 = cpneuron(self.layer1, self.weights2, 'AND')
			self.output = cpneuron(self.weights2, self.weights3, 'AND')

	def softmax(X):
    	exps = np.exp(X)
    	return exps / np.sum(exps)

	def cross_entropy(X,y):
	    """
	    X is the output from fully connected layer (num_examples x num_classes)
	    y is labels (num_examples x 1)
	    	Note that y is not one-hot encoded vector. 
	    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
	    """
	    m = y.shape[0]
	    p = softmax(X)
	    # We use multidimensional array indexing to extract 
	    # softmax probability of the correct label for each sample
	    log_likelihood = -np.log(p[range(m),y])
	    loss = np.sum(log_likelihood) / m
	    return loss

	def mse(x, y):

		"""
		x being the output of the neural network
		y being the actual output for the corresponding input or the label in case of classification
		"""
		se = 0
		for i in range(len(y)):
			error = (y[i] - x[i])**2
			se += error

		mse = se/len(se)

		return mse

	def absolute error(x, Y):
		
		"""
		x being the output of the neural network
		y being the actual output for the corresponding input or the label in case of classification
		"""
		error = 0

		for i in range(len(y)):

			error += abs(y[i] - x[i])

		abs_error = error / len(y)

		return abs_error

			



