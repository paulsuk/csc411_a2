import cPickle
import math
import matplotlib.cbook as cbook
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import os
import pdb
from scipy.ndimage import filters
from scipy.misc import imread
from scipy.misc import imresize
from scipy.io import loadmat
import time
import urllib


############## GIVEN METHODS #################
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return np.exp(y)/np.repeat(np.sum(np.exp(y), axis=1).reshape(y.shape[0], 1), y.shape[1], axis=1)
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return np.tanh(np.dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = np.dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  np.dot(L0, dCdL1.T ) 


############# CLASSES ################
class NeuralNet(object):

	def __init__(self, weights, biases, x, y, test_x=[], test_y=[]):
		'''
		weights are m x 10
		biases are n x 10

		where m is the number of features
		n is the number of images
		'''
		self.weights = weights
		self.biases = biases
		self.x = x
		self.y = y
		self.test_x = test_x
		self.test_y = test_y

	@classmethod
	def random_weights(cls, mean, std, input_size, output_size, x, y, test_x=[], test_y=[]):
		weights = np.random.normal(mean, std, size=(input_size, output_size))
		biases = np.random.normal(mean, std, size=(output_size, 1))
		return cls(weights, biases, x, y, test_x=test_x, test_y=test_y)
	
	def forward_step(self, x):
		out = linear_comb(x, self.weights, self.biases)
		soft = softmax(out)
		return {'out': out, 'softmax': soft}
	
	def finite_diff(self, step=0.000001):
		''' performs finite differences of gradient of  neg log loss cost function with stepsize
		prints the maximum relative error
		'''
		x = self.x
		y = self.y
		b = self.biases
		w = self.weights
		# Output of the softmax
		p = self.forward_step(x)['softmax']
		
		max_e = 0
		exact = dfdw(y, p, x)
		total_err = 0

		for i in range(w.shape[0]):
			for j in range(w.shape[1]):
				w_plus = np.copy(w)
				w_plus[i][j] += step

				o_plus = linear_comb(x, w_plus, b)
				p_plus = softmax(o_plus)

				numerical = (self.cost(y, p_plus) - self.cost(y, p)) / step

				#pdb.set_trace()
				rel_e = abs(exact[i][j] - numerical)
				total_err += rel_e
				if rel_e > max_e:
					max_e = rel_e
		
		print("step size: {}, maximum relative error: {}, total error: {}".format(step, max_e, total_err))

	def train(self, alpha=0.01, max_iter=2000, epsilon=0.0000001):
		x = self.x
		y = self.y
		test_x = self.test_x
		test_y = self.test_y
		b = self.biases
		w = self.weights
		i = 0

		training_perf = []
		testing_perf = []
		training_costs = []
		testing_costs = []

		while i < max_iter:
			p = self.forward_step(x)['softmax']

			b -= alpha * self.dfdb(y, p).mean(axis=0)
			w -= alpha * self.dfdw(y, p, x)

			test_p = self.forward_step(test_x)['softmax']

			train_corr, train_tot = self.evaluate(x, y)
			test_corr, test_tot = self.evaluate(test_x, test_y)
			
			train_perf = train_corr/train_tot
			test_perf = test_corr/test_tot

			training_perf.append(train_perf)
			testing_perf.append(test_perf)

			training_cost = self.cost(y, p)
			testing_cost = self.cost(test_y, test_p)
			training_costs.append(training_cost)
			testing_costs.append(testing_cost)


			'''
			prev_b, prev_w = np.copy(curr_b), np.copy(curr_w)
			self.biases = curr_b
			self.weights = curr_w

			#pdb.set_trace()
			p = self.forward_step(x)['softmax']
			curr_b -= alpha * dfdb(y, p).mean(axis=0)
			curr_w -= alpha * dfdw(y, p, x)

			out = linear_comb(test_x, self.weights, self.biases)
			test_p = softmax(out) 

			
			'''

			if i % 50 == 0:
				print("Iteration {}, training acc: {}, testing acc: {}".format(i, train_perf, test_perf))
				print("training cost: {}, testing cost: {}".format(training_cost, testing_cost))
			i += 1

		return training_perf, testing_perf, training_costs, testing_costs
	
	def graph_perf(self, training_perf, testing_perf, training_costs, testing_costs):
	#Graph stuff
		plt.figure()
		plt.plot(training_perf, 'r', label="training performance")
		plt.plot(testing_perf, 'b', label="testing performance")
		plt.xlabel("Iteration")
		plt.ylabel("Performance")
		plt.title("performance vs Iterations")
		plt.legend(loc=4)
		plt.savefig("part4_perf_vs_iter.jpg")

		plt.figure()
		plt.plot(training_costs, 'r', label="training cost")
		plt.plot(testing_costs, 'b', label="testing cost")
		plt.xlabel("Iteration")
		plt.ylabel("Cost")
		plt.title("Cost vs Iterations")
		plt.legend()
		plt.savefig("part4_cost_vs_iter.jpg")
	
	def evaluate(self, x, y):
		correct = 0
		total = x.shape[0]

		p = self.forward_step(x)['softmax']

		for i in range(p.shape[0]):
			max_index = np.argsort(p[i])[-1]
			correct +=  y[i][max_index]
		
		return correct, total
	
	def saveWeights(self):
		for i in range(self.weights.shape[1]):
			w = self.weights[:, i]
			w.shape = (28, 28)
			mpimg.imsave("weights_GOOD{}.jpg".format(i), w, cmap=plt.cm.coolwarm)

	def cost(self, y, p):
		# Returns neg log loss of the softmax output of NN
		return -np.sum(np.multiply(y, np.log(p))) / y.shape[0]

	def dfdw(self, y, p, x):
		# gradient of cost function wrt the weights
		return np.dot(self.dodw(x).T, self.dfdo(y, p)) / x.shape[0]

	def dfdb(self, y, p):
		return self.dfdo(y, p).mean(axis=0).reshape((10, 1))
	 
	def dfdo(self, y, p):
		# gradient of cost wrt outputs
		return p - y

	def dodw(self, x):
		# gradient of outputs wrt inputs
		return x

class NeuralNetLinear(NeuralNet):

	def forward_step(self, x):
		soft = linear_comb(x, self.weights, self.biases)
		return {'softmax': soft}

	def cost(self, y, p):
		return np.sum((p - y) ** 2) /(2*y.shape[0])

	def dfdw(self, y, p, x):
		# gradient of cost function wrt the weights
		return np.dot(x.T, self.dfdo(y, p)) / x.shape[0]

	def dfdb(self, y, p):
		return np.mean(self.dfdo(y, p), axis=0).reshape((10, 1))
	 
	def dfdo(self, y, p):
		# gradient of cost wrt outputs
		return p - y

	def dodw(self, x):
		# gradient of outputs wrt inputs
		return x
			

############## UTIL METHODS ###############
def linear_comb(x, w, b):
	'''
	Computes the linear activation function
	'''
	return np.dot(x, w) + np.dot(np.ones((x.shape[0], 1)), b.T)


def getData(size, test=False, noisy = False):
	if test:
		setname = "test"
	else:
		setname = "train"

	raw = loadmat("mnist_all.mat")

	np.random.seed(1)
	num_data = raw[setname + "0"]
	np.random.shuffle(num_data)
	num_data = num_data[0:size]

	labels = np.zeros((num_data.shape[0], 10))
	labels[:, 0] = 1
		 
	for i in range(1, 10):
		temp = raw[setname + str(i)]
		np.random.shuffle(temp)
		temp = temp[0:size]

		labels_temp = np.zeros((temp.shape[0], 10))
		labels_temp[:, i] = 1 

		num_data = np.vstack((num_data, temp))
		labels = np.vstack((labels, labels_temp))

	num_data = num_data/255.

	if noisy:
		num_data += (np.random.rand(num_data.shape[0], num_data.shape[1]) - 0.5) * 2
	
	return num_data, labels

############## CODE TO RUN THE PARTS #####################
def part1():
	'''
	Sample the dataset of 10 images per number
	'''
	data = loadmat("mnist_all.mat")
	np.random.seed(8008)
	for i in range(10):
		# For each digit, choose 10 images
		plt.figure()
		num_data = data["train{}".format(i)]
		np.random.shuffle(num_data)

		for j in range(10):
			#x, y = int(j/2), j%2
			plt.subplot(5, 2, j+1)
			#pdb.set_trace()
			frame = plt.gca()
			frame.axes.get_xaxis().set_visible(False)
			frame.axes.get_yaxis().set_visible(False)
			plt.imshow(num_data[j].reshape((28, 28)), cmap=cm.gray)
		plt.savefig("{}s_example.png".format(i))

def part2():
	'''
	Generates a simple neural net with random weights and biases and computes the softmax
	output of the network
	'''
	# One image per digit
	x, y = getData(1)
	net = NeuralNet.random_weights(0, 0.1, x.shape[1], 10, x, y)
	res = net.forward_step(x)
	print(res['softmax'])

def part3():
	'''
	Returns the maximum error between numerical derivative and our gradient formula
	'''
	x, y = getData(150)
	net = NeuralNet.random_weights(0, 0.1, x.shape[1], 10, x, y)
	#pdb.set_trace()
	net.finite_diff()

def part4():
	x, y = getData(100)
	test_x, test_y = getData(100, test=True)
	net = NeuralNet.random_weights(0, 0.1, x.shape[1], 10, x, y, test_x, test_y)
	training_perf, testing_perf, training_costs, testing_costs = net.train()
	net.graph_perf(training_perf, testing_perf, training_costs, testing_costs)
	net.saveWeights()

def part5():
	x, y = getData(100, noisy=True)
	test_x, test_y = getData(100, test=True, noisy=True)
	net = NeuralNet.random_weights(0, 0.1, x.shape[1], 10, x, y, test_x, test_y)
	linear = NeuralNetLinear.random_weights(0, 0.1, x.shape[1], 10, x, y, test_x, test_y)

	print("training softmax")
	train_p, test_p, train_c, test_c = net.train()
	print("training linear")
	train_p_l, test_p_l, train_c_l, test_c_l = linear.train()

	plt.figure()
	plt.plot(train_p, 'r', label="soft train")
	plt.plot(test_p, 'b', label="soft test")
	plt.plot(train_p_l, 'g', label="lin train")
	plt.plot(test_p_l, 'k', label="lin test")
	plt.xlabel("Iteration")
	plt.ylabel("Performance")
	plt.title("Performance vs Iterations with noise")
	plt.legend(loc=4)
	plt.savefig("part5_perf_vs_iter.jpg")

	plt.figure()
	plt.plot(train_c, 'r', label="softmax training cost")
	plt.plot(test_c, 'b', label="softmax testing cost")
	plt.plot(train_c_l, 'g', label="linear training cost")
	plt.plot(test_c_l, 'k', label="linear testing cost")
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.title("Cost vs Iterations with noise")
	plt.legend()
	plt.savefig("part5_cost_vs_iter.jpg")




'''
#Load the MNIST digit data
M = loadmat("mnist_all.mat")

pdb.set_trace()
#Display the 150-th "5" digit from the training set
plt.imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
plt.show()
'''



'''
#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
'''

if __name__ == '__main__':
	# part1()
	# part2()
	# part3()
	# part4()
	part5()
