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
    One or more hidden layers, each with tanh activation ->
    Output layer (vocab-size units) with softmax activation

    As with the MLP, we will keep our bias weights separate from our feature weights to simplify computations.
    '''

    def __init__(self, num_input_units, num_hidden_units, num_layers, num_output_units, char_to_idx, idx_to_char):
        '''Constructor to build the model structure and intialize the weights. There are 3 types of layer:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input units. Equal to vocab size (one-hot encoding).
        num_hidden_units: int. Num units per hidden layer.
        num_layers: int. Num hidden layers.
        num_output_units: int. Num output units. Equal to # data classes.
        char_to_idx: dictionary. Map characters to indices.
        idx_to_char: dictionary. Map indices to characters.

        You do not need to modify this code.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.num_output_units = num_output_units
        self.char_to_ix = char_to_idx
        self.ix_to_char = idx_to_char

        self.initialize_wts()

    def initialize_wts(self, std=0.1):
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        Steps:
        -----
        1. Initialize self.xh_wts, self.hh_wts, self.h_b and self.hq_wts, self.q_b
        with the appropriate sizes according to the normal distribution with standard deviation
        `std` and mean of 0. Use self.num_input_units, self.num_hidden_units, self.num_layers 
        and self.num_output_units as appropriate. NB: self.xh_wts, self.hh_wts, self.h_b will be lists of matrices,
        one for each hidden layer.
        2. Initialize self.mw_xh, self.mw_hh, self.m_bh, self.mw_hq, self.m_bq to zeros. These are used to accumulate information 
        by Adagrad. NB: some of these will be lists of matrices, one for each hidden layer.
        3. Initialize the loss and the running loss.
        ''' 
        # keep the random seed for debugging/test code purposes
        np.random.seed(0)
        
        self.loss = -np.log(1.0/self.num_input_units) #*self.seq_length # loss at iteration 0

        self.running_loss = []

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. Int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        pass

    def predict(self, hprev, seed_ix, n):
        ''' Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        hprev: ndarray. Previous hidden state
        seed_ix: int. Seed letter for first time step
        n: int. Number of characters to generate

        Returns:
        -----------
        y_pred: ndarray. shape=(n,).
            This is the int-coded predicted next characters. 

        Steps:
        -----
        1. Initialize. This is done for you.
        2. Predict! use tanh as the activation function on each hidden layer.
        '''
        ## a one-hot vector
        x = np.zeros((self.num_input_units, 1))
        x[seed_ix] = 1
        ixes = []
        hs = {}
        hs[-1] = np.copy(hprev)

        for t in range(n):
            # predict!

            x[ix] = 1
            ixes.append(ix)

        return ixes

    def forward(self, inputs, ys, hprev):
        '''
        Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity
        to the output layer, ending with the cross-entropy loss computation.
        Don't forget to add the regularization to the loss!

        NOTE: Implement all forward computations within this function
        (don't divide up into separate functions for net_in, net_act). Doing this all in one method
        is not good design, but as you will discover, having the
        forward computations (y_net_in, y_net_act, etc) easily accessible in one place makes the
        backward pass a lot easier to track during implementation. In future projects, we will
        rely on better OO design.

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        inputs: ndarray. Input sequence
        y: ndarray. Int coded class labels
        hprev: ndarray. Previous hidden state

        Returns:
        -----------
        xs: dict. shape=(N, H). Encoded inputs
        ps: dict. shape=(N, H). Output probabilities
        hs: dict. shape=(N, C). Previous hidden states
        loss: float. Loss

        Steps:
        -----
        1. Initialize. This is done for you.
        2. Forward for each item in the input sequence. Update the loss.
        3. Return.
        '''

        xs, hs, yhats, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        for t in range(len(inputs)):
            pass

        return xs, ps, hs, loss


    def backward(self, inputs, ys, xs, ps, hs):
        '''
        Performs backpropagation. 

        Parameters:
        -----------
        inputs: ndarray. Input sequence
        ys: ndarray. Int coded class labels
        xs: dict. shape=(N, H). Encoded inputs
        ps: dict. shape=(N, H). Output probabilities
        hs: dict. shape=(N, C). Previous hidden states

        Returns:
        -----------
        dw_xh, dw_hh, d_bh, dw_hy, d_by: The backwards gradients
        (1) hidden wts with respect to input for each hidden layer, 
        (2) hidden weights with respect to previous hidden state for each hidden layer,
        (3) hidden biases for each hidden layer, 
        (4) output weights, 
        (5) output bias

        Steps:
        -----
        1. Initalize. This is done for you.
        2. For each timestep in the sequence, do backprop as we reviewed in class.
        3. Return
        '''
        dw_xh, dw_hh, dw_hq = [np.zeros_like(w) for w in self.xh_wts], [np.zeros_like(w) for w in self.hh_wts], np.zeros_like(self.hq_wts)
        d_bh, d_bq = [np.zeros_like(b) for b in self.h_b], np.zeros_like(self.q_b)
        dhnext = [np.zeros_like(h) for h in hs[0]]

        for t in reversed(range(len(inputs))):
            pass

        return dw_xh, dw_hh, d_bh, dw_hq, d_bq

    def update(self, dw_xh, dw_hh, dw_hq, d_bh, d_bq, lr):
        '''
        Updates parameters using Adagrad.

        Parameters:
        -----------
        dw_xh, dw_hh, dw_hq, d_bh, d_bq: ndarrays. The gradients
        lr: float. The learning rate

        Returns:
        -----------
        Nothing

        Steps:
        -----
        1. For each hidden layer, clip the gradients. Then update instance variables mw_xh, mw_hh, m_bh, xh_wts, hh_wts and h_b.
        2. Clip the gradients for dw_hq and d_bq. Then update instance variables mw_hq, m_bq, hq_wts and q_b. 
        '''
  
         # loop through each layer
        for i in range(self.num_layers):
            pass

    def fit(self, data, num_steps=25, n_epochs=500, lr=0.0001, verbose=2,
            print_every=100):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        data: ndarray. 
            A sequence of char indices
        num_steps: int.
            How much to unroll the RNN
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
        1. For each epoch:
           * prepare inputs. 
           * for each input run forward, backward, update and update the running loss.
           * print periodically.
        '''
        ## iterator counter
        n = 0
        ## data pointer
        p = 0

        for epoch in range(n_epochs):
            # prepare inputs (we're sweeping from left to right in steps self.num_steps long)
            
            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.predict(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print(txt)

            # forward, backward, update, loss

            if n % 100 == 0:
                print('iter %d, loss: %f' % (n, self.loss)) # print progress
  
            p += num_steps # move data pointer
            n += 1 # iteration counter 
        return self.running_loss

