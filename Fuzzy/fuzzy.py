import numpy as np 

class MinMaxFuzzy:

	flag = True
	
	
	def __init__(self, x, y):

		self.input = x
		self.weights1 = np.random.randn(self.input.shape[1], 32) ##Weights for connections from Input to 1st hidden layer
		self.weights2 = np.random.randn(32, 12)  ##Weights for connections from 1st Hidden Layer to 2nd Hidden Layer
		self.weights3 = np.random.randn(12, 1)  ##Weights for connections from 2nd Hidden Layer to 3rd Hidden Layer
		self.y = y
		self.output = np.zeros(self.y.shape)

	@staticmethod
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

		if activation == "OR":

			self.layer1 = MinMaxFuzzy.cpneuron(self.input, self.weights1, 'OR')
			self.layer2 = MinMaxFuzzy.cpneuron(self.layer1, self.weights2, 'OR')
			self.output = MinMaxFuzzy.cpneuron(self.layer2, self.weights3, 'OR')

		elif activation == "AND":
			
			self.layer1 = MinMaxFuzzy.cpneuron(self.input, self.weights1, 'AND')
			self.layer2 = MinMaxFuzzy.cpneuron(self.layer1, self.weights2, 'AND')
			self.output = MinMaxFuzzy.cpneuron(self.weights2, self.weights3, 'AND')


	def backward_propagation(self, alpha = 0.1, eta = 0.9):

		"""We apply Backpropagation to update out weights of all layers
			To do this we use the chain rule

			d_weights(t) = (-eta * d__E(t)) + (alpha * d_weights(t-1))

			Note: Initial values of d_weights(t-1) = 0
		"""
		if(MinMaxFuzzy.flag):
			d_weights_prev_t_3= np.zeros(self.weights3.shape[0], self.weights3.shape[1]) ##2nd Hidden --> Output
			d_weights_prev_t_2= np.zeros(self.weights2.shape[0], self.weights2.shape[1]) ##1st Hidden --> 2nd Hidden
			d_weights_prev_t_1= np.zeros(self.weights1.shape[0], self.weights1.shape[1]) ##Input -->1st Hidden
			MinMaxFuzzy.flag = False

		## Calculating E(t)
		E_t_3 = np.multiply(-(self.y - self.output) * (self.output), self.layer2) ##For Weights 3
		E_t_2 = np.multiply(-(self.y - self.output) * (self.output), self.layer1) ##For Weights 2
		E_t_1 = np.multiply(-(self.y - self.output) * (self.output), self.input)  ##For Weights 1

		##Compute d_weights(t)
		d_weights_t_3 = np.add((-eta * E_t_3), d_weights_prev_t_3)
		d_weights_t_2 =	np.add((-eta * E_t_2), d_weights_prev_t_2)
		d_weights_t_1 = np.add((-eta * E_t_1), d_weights_prev_t_1)

		##Update the Weights using the derived formula
		self.weights3 = np.add(self.weights3, d_weights_t_3)
		self.weights2 = np.add(self.weights2, d_weights_t_2)
		self.weights1 = np.add(self.weights1, d_weights_t_1)

		## Change the Values of d_weight(t-1)
		d_weights_prev_t_3 = d_weights_t_3
		d_weights_prev_t_2 = d_weights_t_2
		d_weights_prev_t_1 = d_weights_t_1

		
class Loss:

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
	    m = y.shape[0]
	    p = Loss.softmax(x)
	    # We use multidimensional array indexing to extract 
	    # softmax probability of the correct label for each sample
	    log_likelihood = -np.log(p[range(m),y])
	    loss = np.sum(log_likelihood) / m
	    return loss
	@staticmethod
	def mse(self, x, y):

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
	@staticmethod
	def absolute_error(self, x, y):
		
		"""
		x being the output of the neural network
		y being the actual output for the corresponding input or the label in case of classification
		"""
		error = 0

		for i in range(len(y)):

			error += abs(y[i] - x[i])

		abs_error = error / len(y)

		return abs_error