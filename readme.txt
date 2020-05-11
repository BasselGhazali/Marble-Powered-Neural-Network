1) To run app.py and get the neural network parameters, run the script from an anaconda compiler.
2) Fill in the required parameters in the window that appears then click "Next" and do the same for the next window.
N.B. To create custom test data, make sure it's a .mat file with 2 variables "data" and "labels".
3) The process may be repeated to change network parameters by clicking the "Build" button after changing the parameters.
4) Once the network is at a satisfactory level of accuracy, the "parameters" file will contain the values of the weights and biases which will be needed by the physical system.
5) Use the "weight", "threshold" and "multiplier" g-code files to print as many components as required. Each print contains all the components needed by that unit. A single print gives the complete unit which can be assembled. The individual pieces' .stl files are also available to create custom prints.
6) Assemble components in layers, with input layer on top, output at the bottom, and intermediate layers in between. Neurons of the same layer go across horizontally (left and right to the viewer) and weights and multipliers stack towards/away from the user.
7) Connect all resets to each other and use 26mm ID flexible PVC hose to connect modules together.
