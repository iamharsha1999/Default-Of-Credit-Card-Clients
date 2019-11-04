import numpy as np

class FuzzyClasssifier:

    def __init__(self, X, Y, no_of_class):

        self.dimensions = retrive_dimensions(X)                                      ## Number of dimensions of the hyperbox
        self.hyperlayer_nodes = 16                                                   ## Number of hyperboxes in the Hidden Layer
        self.hyperlayer = create_hyperlayer(self.hyperlayer_nodes, self.dimensions)  ## Create the Hidden Layer
        self.output = create_output_layer(no_of_class)                               ## Create the Output Layer


    @staticmethod
    def retrive_dimensions(inputp):
        return inputp.shape[0]


    @staticmethod
    def create_output_layer(inputp):
        return np.zeros(inputp)


    @staticmethod
    def create_hyperlayer(no_of_nodes, no_of_dims):

        """
        no_of_nodes => No of Hyperboxes in the Layer
        no_of_dims  => No of Dimensions of the Hyperbox
        """

        hyperlayer = {}
        for i in range(no_of_nodes):

            points = np.random.randint(0, 15, size = 2 * no_of_dims)
            points.sort()
            _min = points[:no_of_dims]
            _max = points[no_of_dims:]
            hyperlayer["node" + str(i+1)] = [_min,_max]

        return hyperlayer



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
