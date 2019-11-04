import numpy as np

class FuzzyClasssifier:

    def __init__(self, X, Y, no_of_class, exp_bound = 0.5):

        self.expansion_boundary = exp_bound                                              ## User defined Expansion Boundary
        self.dimensions = retrive_dimensions(X)                                          ## Number of dimensions of the hyperbox
        self.hyperlayer_nodes = 16                                                       ## Number of hyperboxes in the Hidden Layer
        self.hyperlayer = create_layer(no_of_class)                                      ## Create the Hidden Layer
        self.hyperweights = create_hyperweights(self.hyperlayer_nodes, self.dimensions)  ## Random Initialisation of Weights
        self.output = create_layer(no_of_class)                                          ## Create the Output Layer
        


    @staticmethod
	def normalise(x):
		maxi = np.max(x)
		mini = np.min(x)
		x = (x - mini)/(maxi-mini)
		return x


    @staticmethod
    def retrive_dimensions(inputp):
        return inputp.shape[0]


    @staticmethod
    def create_layer(inputp):
        return np.zeros(inputp)

    @staticmethod
    def create_output_weights(no_of_nodes,no_of_classes):
        weights = np.random.randint


    @staticmethod
    def create_hyperweights(no_of_nodes, no_of_dims):

        """
        no_of_nodes => No of Hyperboxes in the Layer
        no_of_dims  => No of Dimensions of the Hyperbox
        """

        hyperweights = {}
        for i in range(no_of_nodes):

            points = np.random.randn(size = 2 * no_of_dims)
            points.sort()
            _min = points[:no_of_dims]
            _max = points[no_of_dims:]
            hyperweights["node" + str(i+1)] = [_min,_max]

        return hyperweights


    @staticmethod
    def  membership(Ah, gamma, hypernodes):
        n = len(Ah)
        no_of_nodes = hypernodes.shape[0]
        b_j = max(0,1-max(0,gamma*min(1,a_hi - w_ji))) + max(0,1*max(0,gamma*min(1,v_ji - a_hi)))
        sum = 0
        for j in range(no_of_nodes):
            for i in range(n):
                a_hi = Ah[i]
                w_ji = max_hypernodes[j][i]
                v_ji = min_hypernodes[j][i]
                sum += b_j
            sum = sum/(2*n)
            hyperlayer.append(sum)

        return hyperlayer

    @staticmethod
    def hyperbox_expansion(exp_bound,hyperbox,inpt,dims):
        sum = 0
        for i in range(dims):
            sum += (max(hyperbox[1][i],inpt[i]) - min(hyperbox[0][i]))
        if exp_bound > sum:
            for i in range(dims):
                hyperbox[0][i] = min(hyperbox[0][i],inpt[i])
                hyperbox[1][i] = max(hyperbox[1][i],inpt[i])
