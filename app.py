'''
App.py creates a neural network consisting of positive integer weights and negative integer biases
This code gives the network parameters to anyone who wishes to create a marble-powered mechanical neural network,
which are both saved in a .h5 file as well as displayed in the command window
Network parameters are set by user input which is taken using a series of two pop-up windows
To operate this program, run this file from a python compiler
Written by: Bassel Ghazali
'''

# import tensorflow library and functions
import tensorflow as tf
from tensorflow.python.ops.math_ops import round, floor, ceil, cast, less_equal, greater_equal, subtract
from tensorflow.keras import layers, models, constraints, backend as K, initializers, optimizers, losses

# import mathematical functions for array use
import numpy as np

# import loadmat to load in training and testing data
from scipy.io import loadmat

# import GUI library
from tkinter import *
import tkinter.ttk as TTK
from tkinter import messagebox

def isInt(keypress):
    if keypress.isdigit():
        return True
    elif keypress is "":
        return True
    else:
        return False

# contraint class which limits values to negative integers by rounding when a threshold is exceeded
class NegInt (constraints.Constraint):
    def __call__(self, w):
        # create tensors rounding the parameters up, down, and to the closest integer
        upper = floor(w*cast(less_equal(w, 0.), K.floatx()))
        lower = ceil(w*cast(less_equal(w, 0.), K.floatx()))
        rounded = round(w*cast(less_equal(w, 0.), K.floatx()))

        # replace all parameters which have decreased beyond the threshold to the next lowest integer
        new_w = tf.where(greater_equal(tf.subtract(w,rounded), 0.003), lower, w)

        # replace all parameters which have increased beyond the threshold to the next highest integer
        new_w2 = tf.where(less_equal(tf.subtract(new_w,rounded), -0.003), upper, new_w)

        # round the output to make any unchanged values integers which haven't changed from the last iteration
        new_rounded = round(new_w2*cast(less_equal(new_w2, 0.), K.floatx()))
        return new_rounded

# callable class to create first window of two in network creation process
class StartApp:
    def __init__(self, master):
        self.master = master

        # create input to set number of network intermediate layers
        self.int_layers_lbl = Label(self.master, text="Network intermediate layers")
        self.int_layers_lbl.grid(column=0, row=0)
        self.int_layers = Entry(self.master,width=10)
        self.int_layers.grid(column=1, row=0)

        int_validation = window.register(isInt)
        self.int_layers.config(validate="key",validatecommand=(int_validation,'%P'))

        # create radio buttons to set type of input (1D or 2D)
        self.selected = IntVar()
        self.input_label = Label(window, text="Input Shape")
        self.input_label.grid(column=0, row=1)
        self.rad1 = Radiobutton(window,text='1D', value=0, variable=self.selected)
        self.rad2 = Radiobutton(window,text='2D', value=1, variable=self.selected)
        self.rad1.grid(column=1, row=1)
        self.rad2.grid(column=2, row=1)

        # create button which opens the next dialogue box when pressed
        next_button = Button(window, text="Proceed to build", command=self.window_one)
        next_button.grid(column=2, row=3)

    # create the second window of two in the network creation process and pass the variables from the first to the second
    def window_one(self):
        # extract the variables to be passed into the second window
        int_layers = int(self.int_layers.get())
        input_shape = self.selected.get()

        # create the second window
        self.window_one = Toplevel(self.master)
        window.title("Network Details")
        self.app = SetTraining(self.window_one, int_layers, input_shape)

# callable class to create second window of two in the network then build the network
class SetTraining():
    def __init__(self, master, int_layers, input_shape):
        self.master = master
        self.input_size = []
        self.int_width = []
        self.int_func = []

        # register user entry validation functions using variables recognised by tkinter
        int_validation = window.register(isInt)

        # set the entry widgets for network input based on its dimension
        if input_shape==0:
            self.input_size_lbl = Label(self.master, text="Input size")
            self.input_size_lbl.grid(column=0, row=0)
            self.input_size = Entry(self.master,width=10)
            self.input_size.grid(column=1, row=0)
            self.input_size.config(validate="key",validatecommand=(int_validation,'%P'))
        else:
            self.input_width_lbl = Label(self.master, text="Input width")
            self.input_width_lbl.grid(column=0, row=0)
            self.input_size.append(Entry(self.master,width=10))
            self.input_size[-1].grid(column=1, row=0)
            self.input_size[-1].config(validate="key",validatecommand=(int_validation,'%P'))
            self.input_height_lbl = Label(self.master, text="Input height")
            self.input_height_lbl.grid(column=2, row=0)
            self.input_size.append(Entry(self.master,width=10))
            self.input_size[-1].grid(column=3, row=0)
            self.input_size[-1].config(validate="key",validatecommand=(int_validation,'%P'))

        # create entry widgets to set number of neurons per intermediate layer
        for i in range(1,int_layers+1):
            self.int_width_lbl = Label(self.master, text=("Neurons in intermediate layer " + str(i)))
            self.int_width_lbl.grid(column=0, row=i)
            self.int_width.append(Entry(self.master,width=10))
            self.int_width[-1].grid(column=1, row=i)
            self.int_width[-1].config(validate="key",validatecommand=(int_validation,'%P'))

        # create entry widget to set number of neurons at output
        self.out_width_lbl = Label(self.master, text=("Neurons  at output"))
        self.out_width_lbl.grid(column=0, row=int_layers+2)
        self.out_width = Entry(self.master,width=10)
        self.out_width.grid(column=1, row=int_layers+2)
        self.out_width.config(validate="key",validatecommand=(int_validation,'%P'))

        # place a gap in the dialogue box between neuron parameters and training parameters
        self.gap1 = Label(self.master, text=(""))
        self.gap1.grid(column=0, row=int_layers+3)

        # create entry widget to set training learning rate
        self.lr_lbl = Label(self.master, text="Learning rate")
        self.lr_lbl.grid(column=0, row=int_layers+4)
        self.lr = Entry(self.master,width=10)
        self.lr.grid(column=1, row=int_layers+4)

        # create entry widget to set number of training iterations
        self.epochs_lbl = Label(self.master, text="Training epochs")
        self.epochs_lbl.grid(column=2, row=int_layers+4)
        self.epochs = Entry(self.master,width=10)
        self.epochs.grid(column=3, row=int_layers+4)
        self.epochs.config(validate="key",validatecommand=(int_validation,'%P'))

        # create entry widget to put in training dataset
        self.data_lbl = Label(self.master, text="Training data file name")
        self.data_lbl.grid(column=0, row=int_layers+5)
        self.dataset = Entry(self.master,width=10)
        self.dataset.grid(column=1, row=int_layers+5)

        # place a gap in the dialogue box between training parameters and run button
        self.gap3 = Label(self.master, text=(""))
        self.gap3.grid(column=0, row=int_layers+6)

        # create button to run the network-building function
        self.run_button = Button(self.master, text="Build & Train", command=self.build_network)
        self.run_button.grid(column=3, row=int_layers+7)

    def build_network(self):
        # extract network input and intermediate dimensions
        input_dim = len(self.input_size)
        int_layers = len(self.int_width)

        # initialize neural network
        network = models.Sequential()

        if input_dim==1:
            # create an input layer with the specified number of neurons
            input_length = int(self.input_size.get())
            network.add(layers.Flatten(input_shape=(input_length,)))
        else:
            # take the given input grid and flatten to feed into network input layer
            input_width = int(self.input_size[0].get())
            input_height =int(self.input_size[1].get())
            network.add(layers.Flatten(input_shape=(input_height, input_width)))

        # add as many intermediate layers to network as specified, using integer biases and positive weights restrictions
        for i in range(int_layers):
            layer_width = self.int_width[i].get()
            network.add(layers.Dense(layer_width, activation='hard_sigmoid',kernel_constraint ='NonNeg',bias_constraint = NegInt(), bias_initializer= initializers.RandomUniform(minval=-100., maxval=0., seed=None), kernel_initializer= initializers.RandomUniform(minval=0., maxval=10., seed=None)))

        # add the output layer to the network
        output_width = self.out_width.get()
        network.add(layers.Dense(output_width, activation = 'softmax', kernel_constraint ='NonNeg',bias_constraint = NegInt(), bias_initializer= initializers.RandomUniform(minval=-100., maxval=0., seed=None), kernel_initializer= initializers.RandomUniform(minval=0., maxval=10., seed=None)))

        # extract the training parameters from the user entry and cast them to the correct types
        lr = float(self.lr.get())
        custom_epochs = int(self.epochs.get())

        # create the optimiser function using the specified learning rate
        user_optim = tf.keras.optimizers.SGD(lr=lr)

        network.compile(
        # set the network training method to adaptive moment estimation
        optimizer=user_optim,
        # set the error calculation method to categorical crossentropy
        loss='categorical_crossentropy',
        # display network accuracy during training and testing
        metrics=['categorical_accuracy']
        )

        # extract training data and labels from mat file
        #data = loadmat('dataset.mat')
        data = loadmat(self.dataset.get()+'.mat')
        train_data = data['data']
        train_labels = data['labels']
        train_labels_binary= tf.keras.utils.to_categorical(train_labels)

        # train the network using the training data
        network.fit(train_data, train_labels_binary, epochs=custom_epochs)

        #  extract weights into a variable and round them to the nearest integer
        weights = np.array(network.get_weights(), dtype=object)
        for i in range(np.size(weights)):
            weights[i] = weights[i].round()

        # update the network with the rounded weights and test its performance
        network.set_weights(weights)
        network.evaluate(train_data, train_labels_binary, verbose=2)

        # export network tunable parameters
        network.save_weights('parameters.h5')

'''
APPLICATION ENTRY POINT
Opens the first window to start the network creation process
'''
window = Tk()
window.title("Neural Network Builder")
app = StartApp(window)
window.mainloop()
