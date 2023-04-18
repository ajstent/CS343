'''rnn.py
Constructs, trains, tests recurrent neural network
YOUR NAMES HERE
CS343: Neural Networks
Project 4: Recurrent Neural Networks
'''
import numpy as np

class RNN:
    '''
    RNN is a class for a recurrent neural network.

    The structure of our RNN will be:

    Input layer (vocab-size units) for one-hot encoding of each input character ->
    Hidden layer (1 units) with Rectified Linear activation (ReLu) ->
    Output layer (vocab-size units) with softmax activation

    As with the MLP, we will keep our bias weights separate from our feature weights to simplify computations.
    '''

    def __init__(self, num_input_units, num_hidden_units, num_output_units):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input units. Equal to vocab size (one-hot encoding).
        num_hidden_units: int. Num hidden units.
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        # loss history
        self.loss_history = []

        self.initialize_wts(num_input_units, num_hidden_units, num_output_units)

    def initialize_wts(self, V, H, C, std=0.1):
        ''' Randomly initialize the hidden and output layer weights and bias terms

        Parameters:
        -----------
        V: int. Vocab size (for one hot encoding of inputs).
        H: int. Num hidden units.
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.w_xh, self.w_hh, self.b_h and self.w_hq, self.b_q
        with the appropriate size according to the normal distribution with standard deviation
        `std` and mean of 0.
        '''
        # keep the random seed for debugging/test code purposes
        np.random.seed(0)
        pass

    def one_hot(self, x):
        '''One-hot encode input x."

        Parameters:
        -----------
        x: ndarray. indices of a sequence of items in a vocabulary.

        Returns:
        -----------
        x_one_hot: One-hot encoded inputs.
            e.g. if x = [0, 2, 1] and self.num_input_units = 4 then we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        pass

    def predict(self, h, seed_ix, n):
        ''' Predicts the int-coded class values (next items) given a seed item and a number of items to predict.

        Parameters:
        -----------
        seed_ix: int. seed item for first time step.
        n: int. number of items to generate.

        Returns:
        -----------
        y_pred: ndarray. shape=(n,).
            This is the int-coded predicted next items. 

        Steps:
        -----
        1. One-hot encode seed_ix as x_one_hot
        2. For time step t where t >= 0 and t < n, do a forward pass through the RNN, get the output, and make the one hot encoding of that output the new x_one_hot.
        '''
        pass

    def forward(self, inputs, ys, hprev):
        '''
        Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity
        to the output layer, ending with the cross-entropy loss computation.

        Parameters:
        -----------
        inputs: ndarray. int-encoded inputs. shape=(self.num_steps N, self.num_input_units M)
        ys: ndarray. int coded class labels. shape=(self.num_steps N,)
        hprev: previous hidden state.

        Returns:
        -----------
        xs: dict of size N of ndarrays. Each value has shape (M,1), representing a one-hot encodings of inputs.
        ps: dict of size N of ndarrays. Each value has shape shape=(self.num_output_units C,), representing the probability of each output node.
        hs: dict of size N of ndarrays. Each value has shape shape=(self.num_hidden_units H,), representing the hidden state at each time step.
        loss: float. loss derived from output layer, summed over all input samples.

        Steps:
        -----
        1. For time step t where t >= 0 and t < N:
           a. One-hot-encode inputs[t]
           b. Pass it up through the RNN, get the output
           c. Add the cross entropy loss for this input to the loss
       

        '''
        xs, hs, yhats, ps = {}, {}, {}, {}
        ## record each hidden state of
        hs[-1] = np.copy(hprev)
        loss = 0
        # your code here

        return xs, ps, hs, loss


    def backward(self, inputs, ys, xs, ps, hs):
        '''
        Performs a backward pass (output -> hidden -> input) during training to update the
        weights. This function implements backpropogation through time.

        Parameters:
        -----------
        inputs: ndarray. int-encoded inputs. shape=(self.num_steps N, self.num_input_units M)
        ys: ndarray. int coded class labels. shape=(self.num_steps N,)
        xs: dict of size N of ndarrays. Each value has shape (M,1), representing a one-hot encodings of inputs.
        ps: dict of size N of ndarrays. Each value has shape shape=(self.num_output_units C,), representing the probability of each output node.
        hs: dict of size N of ndarrays. Each value has shape shape=(self.num_hidden_units H,), representing the hidden state at each time step.

        Returns:
        -----------
        dw_xh, dw_hh, db_h, dw_hq, db_q: The backwards gradients
        (1) hidden wts with respect to input, (2) hidden weights with respect to previous hidden state,
        (3) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        Steps:
        -----
        1. Initialize the return values
        2. For time step t over the inputs:
          a. backprop into y; calculate derivative with respect to w_hq and b_q
          c. backprop into h; calculate derivate with respect to w_hh, w_xh and b_h
        3. Clip the gradients
        '''
        dWxh, dWhh, dWhy = np.zeros_like(self.xh_wts), np.zeros_like(self.hh_wts), np.zeros_like(self.hy_wts)
        dbh, dby = np.zeros_like(self.h_b), np.zeros_like(self.y_b)
        dhnext = np.zeros_like(hs[0])
        # Your code here

        return dw_xh, dw_hh, db_h, dw_hq, db_q

  

    def fit(self, data, num_steps=25, resume_training=False, n_epochs=500, lr=0.0001, verbose=2,
            print_every=100):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        data: ndarray. Of variable length.
        resume_training: bool.
            False: we clear the network weights and do fresh training
            True: we continue training based on the previous state of the network.
                This is handy if runs of training get interupted and you'd like to continue later.
        num_steps: int.
            Maximum number of steps to unroll the network to
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        verbose: int.
            0 means no print outs. Any value > 0 prints Current epoch number and training loss every
            `print_every` (e.g. 100) epochs.
        print_every: int.
            If verbose > 0, print out the training loss and validation accuracy over the last epoch
            every `print_every` epochs.
            Example: If there are 20 epochs and `print_every` = 5 then you print-outs happen on
            on epochs 0, 5, 10, and 15 (or 1, 6, 11, and 16 if counting from 1).

        Returns:
        -----------
        loss_history: Python list of floats.
            Recorded training loss on every epoch for the current mini-batch.

        Steps:
        -----
        1. In the main training loop, sample to get a sequence of no longer than num_steps.
        2. Do forward pass through network using the sequence.
        3. Do backward pass through network using the sequence.
        4. Add the loss so far to our loss history list.
        7. Use the Python time module to print out the runtime (in minutes) for iteration 0 only.
            Also printout the projected time for completing ALL training iterations.

        Implement Adam as the optimizer.
        '''
        pass

