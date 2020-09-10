'''
    Feed-Foward Artificial Neural Network
    -------------------------------------

    Feed-foward nets are the simplest NN's to master. They are comprised of
    an input layer, one or more hidden layers and an output layer. Since the
    dimensionality of out data is 2 inputs to 1 output, there will be 2 input
    neruons and a single output neuron. For sake of simplicty this net will
    restrict itself to a single hidden layer (deep belief networks can be for
    another time).

    This model revolves around on estimating your SAT score based on the amount of
    hours you slept and the amount of hours you studied the night before. For
    more information including the theory papers for the algorthms behind the
    backpropagation refer to the user manual.

    This software does requires 2 dependencies:
      > Numpy Library (https://docs.scipy.org/doc/numpy-1.13.0/user/install.html)
      > Scipy Library (https://www.scipy.org/install.html)

    Python Version: 3.6

    28.07.2017 | Oakhill College | SDD | Open Source Software (C) | Lucas Barbosa
'''

# dependencies for operation
import sys
import numpy as np
from scipy import optimize

class Neural_Network(object):

    def __init__(self, learning_rate=0):
        # define hyperparameters
        self.input_layer_size = 2
        self.hidden_layer_size = 3
        self.output_layer_size = 1

        # define parameters
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

        # regularization parameter
        self.learning_rate = learning_rate

    # forward propagation
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        prediction = self.sigmoid(self.z3)
        return prediction

    # activation functions
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # derivative of sigmoid function
    def sigmoid_prime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    # efficient backprop
    def cost_function(self, X, desired_output):
        self.prediction = self.forward(X)
        total_error = ((1/2) * sum((desired_output - self.prediction)**2)) / X.shape[0] + \
                      (self.learning_rate / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return total_error

    def cost_function_prime(self, X, desired_y):
        self.prediction = self.forward(X)

        # layer 3 backprop error
        l3_backprop_error   = np.multiply(-(desired_y - self.prediction), \
                              self.sigmoid_prime(self.z3))
        # divide by X.shape[0] to account for the scale of the data
        cost_in_terms_of_W2 = np.dot(self.a2.T, l3_backprop_error) / X.shape[0] + \
                              (self.learning_rate * self.W2)

        # layer 2 backprop error
        l2_backprop_error   = np.dot(l3_backprop_error, self.W2.T) * \
                              self.sigmoid_prime(self.z2)
        # divide by X.shape[0] to account for the scale of the data
        cost_in_terms_of_W1 = np.dot(X.T, l2_backprop_error) / X.shape[0] + \
                              (self.learning_rate * self.W1)

        return cost_in_terms_of_W1, cost_in_terms_of_W2

    # altering and setting the parameters during training
    def get_params(self):
        # get W1 & W2 rolled into a vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        # set W1 & W2 using single parameter vector
        W1_start = 0
        W1_end   = self.hidden_layer_size * self.input_layer_size
        # reshape the W1 weights
        self.W1  = np.reshape(params[W1_start : W1_end], \
                   (self.input_layer_size, self.hidden_layer_size))
        W2_end   = W1_end + self.hidden_layer_size * self.output_layer_size
        # reshape the W2 weights
        self.W2  = np.reshape(params[W1_end : W2_end], \
                   (self.hidden_layer_size, self.output_layer_size))

    def compute_gradient(self, X, desired_y):
        cost_in_terms_of_W1, cost_in_terms_of_W2 = self.cost_function_prime(X, desired_y)
        return np.concatenate((cost_in_terms_of_W1.ravel(), cost_in_terms_of_W2.ravel()))

class Helper(object):

    def __init__(self, Local_Ref):
        # set a local reference to NN class
        self.Local_Ref = Local_Ref

    # normalize data to account for different units
    def scale_data(self, hours, test_score):
        MAX_SCORE = 100.
        hours      /= np.amax(hours, axis=0)
        test_score /= MAX_SCORE
        return hours, test_score

    # print out the results of the NN's predicitons
    def print_predictions(self, train_x, train_y):
        print("="*50)
        print("Expected Scores:")
        for i in range(0, len(train_y)):
            print(int(train_y[i] * 100), "/100", sep="")

        print("="*50)

        predictions = NN.forward(train_x)
        print("Predicted Scores:")
        for i in range(0, len(train_x)):
            print(int(predictions[i] * 100), "/100", sep="")
        print("="*50)

    # checking gradients with numerical gradient computation avoiding logic errors
    def compute_numerical_gradient(self, X, desired_y):
        initial_params     = self.Local_Ref.get_params()
        numerical_gradient = np.zeros(initial_params.shape)
        perturb            = np.zeros(initial_params.shape)

        # epsilon value needs to be small enough act as a 'zero'
        epsilon = 1e-4

        for i in range(len(initial_params)):
            # set perturbation vector to alter the original state of the initial params
            perturb[i] = epsilon
            self.Local_Ref.set_params(initial_params + perturb)
            loss_2 = self.Local_Ref.cost_function(X, desired_y)

            self.Local_Ref.set_params(initial_params - perturb)
            loss_1 = self.Local_Ref.cost_function(X, desired_y)

            # computer numerical gradient
            numerical_gradient[i] = (loss_2 - loss_1) / (2 * epsilon)

            perturb[i] = 0

        self.Local_Ref.set_params(initial_params)
        return numerical_gradient

class Trainer(object):

    def __init__(self, Local_Ref):
        # make local reference to NN
        self.Local_Ref = Local_Ref

    def cost_function_wrapper(self, params, X, desired_y):
        self.Local_Ref.set_params(params)
        total_cost = self.Local_Ref.cost_function(X, desired_y)
        gradient   = self.Local_Ref.compute_gradient(X, desired_y)
        return total_cost, gradient

    # track cost function value as training progresses
    def callback(self, params):
        self.Local_Ref.set_params(params)
        self.cost_list.append(self.Local_Ref.cost_function(self.train_x, self.train_y))
        self.test_cost_list.append(self.Local_Ref.cost_function(self.test_x, self.test_y))

    def train(self, train_x, train_y, test_x, test_y):

        # internal variable for callback function
        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        self.test_y = test_y

        # empty lists to store costs
        self.cost_list = []
        self.test_cost_list = []

        initial_params =  self.Local_Ref.get_params()

        # using scipy's built in Quasi-Newton BFGS mathematical optimization algorithm
        options = {"maxiter": 200, "disp": True}
        _result = optimize.minimize(self.cost_function_wrapper, initial_params, jac=True, \
                                    method="BFGS", args=(train_x, train_y), options=options, \
                                    callback=self.callback)

        # once the training is complete finally set the new values of the parameters in
        self.Local_Ref.set_params(_result.x)
        self.optimization_results = _result

if __name__ == "__main__":

    # check if numpy and scipy are installed before running any code
    if "numpy" not in sys.modules or "scipy" not in sys.modules:
        raise AssertionError("The required dependencies have not been imported.")

    # training data
    train_x = np.array(([3,5],[5,1],[10,2],[6,1.5]), dtype=float)
    train_y = np.array(([75],[82],[93],[70]), dtype=float)

    # testing data
    test_x = np.array(([4, 5.5],[4.5, 1],[9,2.5],[6,2]), dtype=float)
    test_y = np.array(([70],[89],[85],[75]), dtype=float)

    # initialize all the classes
    NN = Neural_Network(learning_rate=0.0001)
    Aux = Helper(NN)
    T1 = Trainer(NN)

    # normalize data
    train_x, train_y = Aux.scale_data(train_x, train_y)
    test_x, test_y   = Aux.scale_data(test_x, test_y)

    # check to see gradients have been correctly calculated
    numerical_gradient = Aux.compute_numerical_gradient(train_x, train_y)
    computed_gradient  = NN.compute_gradient(train_x, train_y)

    # train the network
    T1.train(train_x, train_y, test_x, test_y)

    # observe the results of the tests on above datasets
    Aux.print_predictions(train_x, train_y)
