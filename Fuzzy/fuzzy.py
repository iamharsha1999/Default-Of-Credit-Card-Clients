import numpy as np
import sys
from sklearn.utils import shuffle
import math


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
		print("Layer 1: {}".format(16))
		print("Layer 2: {}".format(8))
		# print("Layer 3: {}".format(8))
		print("Output Layer: {}".format(1))
		self.weights1 = np.random.randint(0, 10 , size = (16,self.input.shape[1])) ##Weights for connections from Input to 1st hidden layer
		self.weights1 = np.around(MinMaxFuzzy.normalise(self.weights1), decimals = 2)
		# self.weights2 = np.random.randint(0, 10 , size = (8,16))  ##Weights for connections from 1st Hidden Layer to 2nd Hidden Layer
		# self.weights2 = np.around(MinMaxFuzzy.normalise(self.weights2), decimals = 2)
		self.weights2 = np.random.randint(0 , 10, size = (8, 16))
		self.weights2 = np.around(MinMaxFuzzy.normalise(self.weights2), decimals  = 2)
		self.weights3 = np.random.randint(0 , 10, size = (1, 8))  ##Weights for connections from 2nd Hidden Layer to 3rd Hidden Layer
		self.weights3 = np.around(MinMaxFuzzy.normalise(self.weights3), decimals = 2)
		self.y = y_d

		self.output = np.zeros(self.y.shape)

		self.declare_prev_weights()

	def declare_prev_weights(self):

		##Declare the weights
		# self.d_weights_prev_t_4 = np.zeros((self.weights4.shape[0], self.weights4.shape[1]))
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
					ncmps = min(ncmps, max(b[j][i], output_previous[0][i]))
				cmps.append(ncmps)

		cmps = np.array(cmps)
		cmps = np.reshape(cmps,(1,cmps.shape[0]))




		return cmps


	def feedforward(self, activation="OR"):

		if activation == "OR":

			self.layer1 = MinMaxFuzzy.cpneuron(self.input, self.weights1, 'OR')
			self.layer2 = MinMaxFuzzy.cpneuron(self.layer1, self.weights2, 'OR')
			# self.layer3 = MinMaxFuzzy.cpneuron(self.layer2, self.weights3, 'OR')
			self.output = MinMaxFuzzy.cpneuron(self.layer2, self.weights3, 'OR')

		elif activation == "AND":

			self.layer1 = MinMaxFuzzy.cpneuron(self.input, self.weights1, 'AND')
			self.layer2 = MinMaxFuzzy.cpneuron(self.layer1, self.weights2, 'AND')
			# self.layer3 = MinMaxFuzzy.cpneuron(self.layer2, self.weights3, 'AND')
			self.output = MinMaxFuzzy.cpneuron(self.layer2, self.weights3, 'AND')


	def backward_propagation(self, batch_loss, alpha = 0.1, eta = 0.4):



		"""We apply Backpropagation to update out weights of all layers
			To do this we use the chain rule

			d_weights(t) = (-eta * d__E(t)) + (alpha * d_weights(t-1))

			Note: Initial values of d_weights(t-1) = 0
		"""
		if self.optimizer == 'normal':
			## Calculating E(t)
			# E_t_4 = np.multiply(-(self.y - self.output) * (self.output), self.layer3) ##For weights 4
			E_t_3 = np.multiply(-(self.y - self.output) * (self.output), self.layer2) ##For Weights 3
			E_t_2 = np.multiply(-(self.y - self.output) * (self.output), self.layer1) ##For Weights 2
			E_t_1 = np.multiply(-(self.y - self.output) * (self.output), self.input)  ##For Weights 1

			##Compute d_weights(t)
			# d_weights_t_4 = np.add((-eta * E_t_4), np.multiply(alpha,self.d_weights_prev_t_4))
			d_weights_t_3 = np.add((-eta * E_t_3), np.multiply(alpha,self.d_weights_prev_t_3))
			d_weights_t_2 =	np.add((-eta * E_t_2), np.multiply(alpha,self.d_weights_prev_t_2))
			d_weights_t_1 = np.add((-eta * E_t_1), np.multiply(alpha,self.d_weights_prev_t_1))

			##Update the Weights using the derived formula
			# self.weights4  = np.add(self.weights4, d_weights_t_4)
			self.weights3 = self.weights3 + d_weights_t_3
			self.weights2 = self.weights2 + d_weights_t_2
			self.weights1 = self.weights1 + d_weights_t_1

			# self.weights1 = MinMaxFuzzy.normalise(self.weights1)
			# self.weights2 = MinMaxFuzzy.normalise(self.weights2)
			# self.weights3 = MinMaxFuzzy.normalise(self.weights3)

			## Change the Values of d_weight(t-1)
			# d_weights_prev_t_4 = d_weights_t_4
			d_weights_prev_t_3 = d_weights_t_3
			d_weights_prev_t_2 = d_weights_t_2
			d_weights_prev_t_1 = d_weights_t_1

		elif self.optimizer == 'sgd':

			d_weights_t_3 = 2 * (batch_loss) * self.layer2
			d_weights_t_2 = 2 * (batch_loss) * np.dot(self.weights3.T,self.layer1)
			d_weights_t_1 = 2 * (batch_loss) * np.dot(np.dot(self.weights3,self.weights2).T,self.input)
			# d_weights_t_1 = 2 * (batch_loss) * np.dot(np.dot(np.dot(self.weights4,self.weights3),self.weights2).T, self.input)

			## Update the Weights
			# self.weights4 = self.weights4 - (alpha * d_weights_t_4)
			self.weights3 = self.weights3 - (alpha * d_weights_t_3)
			self.weights2 = self.weights2 - (alpha * d_weights_t_2)
			self.weights1 = self.weights1 - (alpha * d_weights_t_1)



	def optimize(self,activation, epochs):

			self.feedforward(activation=activation)
			self.backward_propagation()

	@staticmethod
	def cal_acc(y,x):
		"""
		y being the output of the neural
		x being the ground truth
		"""
		correct = 0

		for i in range(len(y)):
			if y[i] == x[i][0][0]:
				correct+=1

		accuracy = (correct/len(y)) * 100

		return accuracy






	def fit(self, x, y, activation, epochs, loss, batch_size = 1024, verbose = 1000):
		y = np.array(y)
		x = np.array(x)

		self.optimzer = loss
		## Caluculate the No of Batches
		if len(x)%batch_size != 0 :
			no_of_batches = int(len(x)/batch_size) + 1
		else:
			no_of_batches  = len(x)/batch_size

		toolbar_width = int(len(x)/verbose)

		print("Number of Epochs: {}".format(epochs))
		print("Activation : {}".format(activation))
		print("Number of Training Examples: {}".format(len(x)))

		for j in range(epochs):
			x,y = shuffle(x, y)
			print("Epoch {}".format(j))
			sys.stdout.write("[%s]" % (" " * toolbar_width))
			sys.stdout.flush()
			sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

			##Initializing the Loss to Zero
			loss = 0

			##Initializing the variables for accuracy
			cal_y = []
			cal_x = []
			batch_start = 0
			batch_end = batch_size

			for batch in range(no_of_batches):

				batch_loss = 0
				d_error = 0
				for i in range(batch_start, batch_end):
					x_d = np.reshape(x[i], (1, x.shape[1]))
					y_d = np.reshape(y[i], (1, 1))
					self.input = x_d
					self.y = y_d
					self.feedforward(activation=activation)
					if self.output > 0.5:
						cal_y.append(1)
					else:
						cal_y.append(0)

					cal_x.append(self.y)

					# Updating the Batch_Loss
					batch_loss += Loss.mse(self.output, self.y)
					d_error = np.add(d_error, (self.output - self.y), casting = 'unsafe')

				## Calculate the MSE
				batch_loss = batch_loss/batch_size
				d_error = (d_error/batch_size)

				## Update the weights
				self.backward_propagation(d_error)
				loss += batch_loss/no_of_batches

				## Update the bar
				sys.stdout.write("=")
				sys.stdout.flush()

				if batch == no_of_batches-2:
					batch_start += len(x)%batch_size
					batch_end += len(x)%batch_size
				else:
					batch_start += batch_size
					batch_end += batch_size

			##Calculate the loss
			loss = loss/len(x)

			##Calculate the accuracy
			accuracy = round(MinMaxFuzzy.cal_acc(cal_y, cal_x),2)

			sys.stdout.write("]\t")
			sys.stdout.write("Accuracy:" + str(accuracy)+ " ")
			print("Loss :{}".format(loss[0][0]))

class Activation:


	def sigmoid(x):
		return 1 / (1 + math.exp(-x))


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
	    m = y
	    p = Loss.softmax(x)
	    # We use multidimensional array indexing to extract
	    # softmax probability of the correct label for each sample
	    log_likelihood = -np.log(p[range(m),y])
	    loss = np.sum(log_likelihood) / m
	    return loss

	@staticmethod
	def mse(x, y):

		"""
		x being the output of the neural network
		y being the actual output for the corresponding input or the label in case of classification
		"""

		error = ((y - x)**2)/len(x)

		return error

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
