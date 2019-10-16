import numpy as np
from numba import jit,cuda
import sys
import time

class MinMaxFuzzy:

	@staticmethod
	def normalise(x):
		maxi = np.max(x)
		mini = np.min(x)
		x = (x - mini)/(maxi-mini)
		return x

	def __init__(self, x, y):

		y = np.array(y)
		x = np.array(x)
		x_d = np.reshape(x[0], (1, x.shape[1]))
		y_d = np.reshape(y[0], (1, 1))
		self.input = x_d

		print("Initialising the model with Input shape.{}".format(x_d.shape))
		print("Layer 1: {}".format(32))
		print("Layer 2: {}".format(12))
		print("Output Layer: {}".format(1))
		self.weights1 = np.random.random((32,self.input.shape[1])) ##Weights for connections from Input to 1st hidden layer
		self.weights1 = MinMaxFuzzy.normalise(self.weights1)
		self.weights2 = np.random.random((12,32))  ##Weights for connections from 1st Hidden Layer to 2nd Hidden Layer
		self.weights2 = MinMaxFuzzy.normalise(self.weights2)
		self.weights3 = np.random.random((1, 12))  ##Weights for connections from 2nd Hidden Layer to 3rd Hidden Layer
		self.weights3 = MinMaxFuzzy.normalise(self.weights3)
		self.y = y_d

		self.output = np.zeros(self.y.shape)
		self.d_weights_prev_t_3 = np.zeros((self.weights3.shape[0], self.weights3.shape[1]))  ##2nd Hidden --> Output
		self.d_weights_prev_t_2 = np.zeros((self.weights2.shape[0], self.weights2.shape[1]))  ##1st Hidden --> 2nd Hidden
		self.d_weights_prev_t_1 = np.zeros((self.weights1.shape[0], self.weights1.shape[1]))  ##Input -->1st Hidden

	@staticmethod

	def cpneuron(output_previous, b, neuronfunction):

		cmps = []
		
		if neuronfunction == 'OR':

			for j in range(b.shape[0]):
				ncmps = 0
				for i in range(output_previous.shape[1]):
					ncmps = max(ncmps, min(b[j][i], output_previous[0][i]))
				cmps.append(ncmps)
		
		elif neuronfunction == 'AND':

			for j in range(b.shape[0]):
				ncmps = 1
				for i in range(output_previous.shape[1]):
					ncmps = min(ncmps, max(b[j], output_previous[i]))
				cmps.append(ncmps)

		cmps = np.array(cmps)
		cmps = np.reshape(cmps,(1,cmps.shape[0]))


	

		return cmps
	# def input_processor(self):
	# 	x = self.input

	def feedforward(self, activation="OR"):

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


		## Calculating E(t)
		E_t_3 = np.multiply(-(self.y - self.output) * (self.output), self.layer2) ##For Weights 3
		E_t_2 = np.multiply(-(self.y - self.output) * (self.output), self.layer1) ##For Weights 2
		E_t_1 = np.multiply(-(self.y - self.output) * (self.output), self.input)  ##For Weights 1

		##Compute d_weights(t)
		d_weights_t_3 = np.add((-eta * E_t_3), np.multiply(alpha,self.d_weights_prev_t_3))
		d_weights_t_2 =	np.add((-eta * E_t_2), np.multiply(alpha,self.d_weights_prev_t_2))
		d_weights_t_1 = np.add((-eta * E_t_1), np.multiply(alpha,self.d_weights_prev_t_1))

		##Update the Weights using the derived formula
		self.weights3 = np.add(self.weights3, d_weights_t_3)
		self.weights2 = np.add(self.weights2, d_weights_t_2)
		self.weights1 = np.add(self.weights1, d_weights_t_1)

		self.weights1 = MinMaxFuzzy.normalise(self.weights1)
		self.weights2 = MinMaxFuzzy.normalise(self.weights2)
		self.weights3 = MinMaxFuzzy.normalise(self.weights3)

		## Change the Values of d_weight(t-1)
		d_weights_prev_t_3 = d_weights_t_3
		d_weights_prev_t_2 = d_weights_t_2
		d_weights_prev_t_1 = d_weights_t_1

		return d_weights_t_3,d_weights_t_2,d_weights_t_1

	def optimize(self,activation, epochs):

			self.feedforward(activation=activation)
			self.backward_propagation()

	def fit(self, x, y, activation, epochs):
		y = np.array(y)
		x = np.array(x)

		toolbar_width = int(len(x)/1000)

		print("Number of Epochs: {}".format(epochs))
		print("Activation : {}".format(activation))
		print("Number of Training Examples: {}".format(len(x)))

		for j in range(epochs):
			print("Epoch {}".format(j))
			sys.stdout.write("[%s]" % (" " * toolbar_width))
			sys.stdout.flush()
			sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
			for i in range(len(x)):
				x_d = np.reshape(x[i], (1, x.shape[1]))
				y_d = np.reshape(y[i], (1, 1))
				self.input = x_d
				self.output = y_d
				self.optimize(activation, epochs)

				# update the bar
				if i%1000 ==0:
					sys.stdout.write("=")
					sys.stdout.flush()
			sys.stdout.write("]\n")



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