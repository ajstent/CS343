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
        num_steps: int. Num of steps to unroll the RNN for.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        self.initialize_wts(num_input_units, num_hidden_units, num_output_units)

    def initialize_wts(self, V, H, C, std=0.1):
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        V: int. Vocab size (for one hot encoding of inputs)
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.xh_wts, self.hh_wts, self.h_b and self.hq_wts, self.q_b
        with the appropriate size according to the normal distribution with standard deviation
        `std` and mean of 0.
        '''
        # keep the random seed for debugging/test code purposes
        np.random.seed(0)
        #self.xh_wts = np.random.randn(H, V)*0.01 # input to hidden
        #self.hh_wts = np.random.randn(H, H)*0.01 # hidden to hidden
        #self.hy_wts = np.random.randn(V, H)*0.01 # hidden to output
        #self.h_b = np.zeros((H, 1)) # hidden bias
        #self.y_b = np.zeros((V, 1)) # output bias


        self.xh_wts = np.random.normal(0, std, (H, V))
        self.hh_wts = np.random.normal(0, std, (H, H))
        self.hq_wts =  np.random.normal(0, std, (C, H))
        self.h_b = np.random.normal(0, std, (H,1))
        self.q_b = np.random.normal(0, std, (C,1))
        # pass

    def accuracy(self, y, y_pred):
        ''' Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        return np.sum(y == y_pred) / len(y)
        # pass

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
        # pass

    def predict(self, h, seed_ix, n):
        ''' Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        seed_ix: int. seed letter for first time step
        n: int. number of characters to generate

        Returns:
        -----------
        y_pred: ndarray. shape=(n,).
            This is the int-coded predicted next characters. 
        '''
        ## a one-hot vector
        x = np.zeros((self.num_input_units, 1))
        x[seed_ix] = 1

        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.xh_wts, x) + np.dot(self.hh_wts, h) + self.h_b)
            #h = np.where(h<=0,0,h)
            y = np.dot(self.hq_wts, h) + self.q_b
            ## softmax
            p = np.exp(y) / np.sum(np.exp(y))
            ## sample according to probability distribution
            ix = np.random.choice(range(self.num_output_units), p=p.ravel())

            ## update input x
            ## use the new sampled result as last input, then predict next char again.
            x = np.zeros((self.num_input_units, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes
        # pass

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
        features: ndarray. net inputs. shape=(mini-batch-size N, Num features M)
        y: ndarray. int coded class labels. shape=(mini-batch-size N,)
        reg: float. regularization strength.

        Returns:
        -----------
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        NOTE:
        - To regularize loss for multiple layers, you add the usual regularization to the loss
          from each set of weights (i.e. 2 in this case).
        '''
        xs, hs, yhats, ps = {}, {}, {}, {}
        ## record each hidden state of
        hs[-1] = np.copy(hprev)
        loss = 0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.num_input_units, 1)) # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
    
            ## hidden state, using previous hidden state hs[t-1]
            hs[t] = np.tanh(np.dot(self.xh_wts, xs[t]) + np.dot(self.hh_wts, hs[t-1]) + self.h_b)
            #hs[t] = np.where(hs[t]<=0,0,hs[t])

            ## unnormalized log probabilities for next chars
            yhats[t] = np.dot(self.hq_wts, hs[t]) + self.q_b
            ## probabilities for next chars, softmax
            ps[t] = np.exp(yhats[t]) / np.sum(np.exp(yhats[t]))
            ## softmax (cross-entropy loss)
            loss += -np.log(ps[t][ys[t], 0])
        return xs, ps, hs, loss


    def backward(self, inputs, ys, xs, ps, hs):
        '''
        Performs a backward pass (output -> hidden -> input) during training to update the
        weights. This function implements the backpropogation algorithm.

        This should start with the loss and progate the activity
        backwards through the net to the input-hidden weights.

        I added dz_net_act for you to start with, which is your cross-entropy loss gradient.
        Next, tackle dz_net_in, dz_wts and so on.

        I suggest numbering your forward flow equations and process each for
        relevant gradients in reverse order until you hit the first set of weights.

        Don't forget to backpropogate the regularization to the weights!
        (I suggest worrying about this last)

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        hidden_state_previous: ndarray. the hidden state at time t-1. shape
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dw_xh, dw_hh, d_bh, dw_hy, d_by: The backwards gradients
        (1) hidden wts with respect to input, (2) hidden weights with respect to previous hidden state,
        (3) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        NOTE:
        - Don't forget to clip gradients.
        '''
        dWxh, dWhh, dWhy = np.zeros_like(self.xh_wts), np.zeros_like(self.hh_wts), np.zeros_like(self.hq_wts)
        dbh, dby = np.zeros_like(self.h_b), np.zeros_like(self.q_b)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            ## compute derivative of error w.r.t the output probabilites
            ## dE/dy[j] = y[j] - t[j]
            dy = np.copy(ps[t])
            dy[ys[t]] -= 1 # backprop into y
    
            ## output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
            ## of output layer. 
            ## then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
            ## dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
    
            ## backprop into h
            ## derivative of error with regard to the output of hidden layer
            ## derivative of H, come from output layer y and also come from H(t+1), the next time H
            dh = np.dot(self.hq_wts.T, dy) + dhnext
            ## backprop through tanh nonlinearity
            ## derivative of error with regard to the input of hidden layer
            ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
    
            ## derivative of the error with regard to the weight between input layer and hidden layer
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            ## derivative of the error with regard to H(t+1)
            ## or derivative of the error of H(t-1) with regard to H(t)
            dhnext = np.dot(self.hh_wts.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dWxh, dWhh, dbh, dWhy, dby

  

    def fit(self, data, num_steps=25,
            resume_training=False, n_epochs=500, lr=0.0001, mini_batch_sz=256, verbose=2,
            print_every=100):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features).
            Features over N inputs.
        y: ndarray.
            int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each
            epoch.
        y_validation: ndarray.
            int-coded class assignments of validation samples. 0,...,numClasses-1
        resume_training: bool.
            False: we clear the network weights and do fresh training
            True: we continue training based on the previous state of the network.
                This is handy if runs of training get interupted and you'd like to continue later.
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        mini_batch_sz: int.
            Batch size per epoch. i.e. How many samples we draw from features to pass through the
            model per training epoch before we do gradient descent and update the wts.
        reg: float.
            Regularization strength used when computing the loss and gradient.
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
        train_acc_history: Python list of floats.
            Recorded accuracy on every training epoch on the current training mini-batch.
        validation_acc_history: Python list of floats.
            Recorded accuracy on every epoch on the validation set.

        TODO:
        -----------
        The flow of this method should follow the one that you wrote in softmax_layer.py.
        The main differences are:
        1) Remember to update weights and biases for all layers!
        2) At the end of an epoch (calculated from iterations), compute the training and
            validation set accuracy. This is only done once every epoch because "peeking" slows
            down the training.
        3) Add helpful printouts showing important stats like num_epochs, num_iter/epoch, num_iter,
        loss, training and validation accuracy, etc, but only if verbose > 0 and consider `print_every`
        to control the frequency of printouts.
        '''
        ## iterator counter
        n = 0
        ## data pointer
        p = 0

        mWxh, mWhh, mWhy = np.zeros_like(self.xh_wts), np.zeros_like(self.hh_wts), np.zeros_like(self.hq_wts)
        mbh, mby = np.zeros_like(self.h_b), np.zeros_like(self.q_b) # memory variables for Adagrad
        smooth_loss = -np.log(1.0/self.num_input_units)*num_steps # loss at iteration 0

        for epoch in range(n_epochs):
            # prepare inputs (we're sweeping from left to right in steps self.num_steps long)
            if p + num_steps + 1 >= len(data) or n == 0:
                # reset RNN memory
                ## hprev is the hiddden state of RNN
                hprev = np.zeros((self.num_hidden_units, 1))
                # go from start of data
                p = 0

            inputs = [self.char_to_ix[ch] for ch in data[p : p + num_steps]]
            ys = [self.char_to_ix[ch] for ch in data[p + 1 : p + num_steps + 1]]

            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.predict(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print('---- sample -----')
                print('----\n %s \n----' % (txt, ))

            # forward self.num_steps characters through the net and fetch gradient
            xs, ps, hs, loss = self.forward(inputs, ys, hprev)
            dWxh, dWhh, dbh, dWhy, dby = self.backward(inputs, ys, xs, ps, hs)
            hprev = hs[len(inputs)-1]
            ## author using Adagrad(a kind of gradient descent)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 100 == 0:
                print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
            # perform parameter update with Adagrad
            ## parameter update for Adagrad is different from gradient descent parameter update
            ## need to learn what is Adagrad exactly is.
            ## seems using weight matrix, derivative of weight matrix and a memory matrix, update memory matrix each iteration
            ## memory is the accumulation of each squared derivatives in each iteration.
            ## mem += dparam * dparam
            for param, dparam, mem in zip([self.xh_wts, self.hh_wts, self.hq_wts, self.h_b, self.q_b],
                                            [dWxh, dWhh, dWhy, dbh, dby],
                                            [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                ## learning_rate is adjusted by mem, if mem is getting bigger, then learning_rate will be small
                ## gradient descent of Adagrad
                param += -lr * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += num_steps # move data pointer
            n += 1 # iteration counter 
        # pass

