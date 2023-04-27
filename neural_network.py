import numpy as np
import pdb

class NeuralNetwork:
    # CREATE Neural Network and set up parameters

    def __init__(self, layer1weights1, layer1weights2, layer1biases1, layer1biases2, layer2weights1, layer2biases1, layer3weights1, layer3weights2, layer3weights3, layer3biases1, layer3biases2, layer3biases3, layer4weights1, layer4weights2, layer4biases1, layer4biases2):
        self.layer1weights1 = layer1weights1
        self.layer1weights2 = layer1weights2
        self.layer1biases1 = layer1biases1
        self.layer1biases2 = layer1biases2

        self.layer2weights1 = layer2weights1
        self.layer2biases1 = layer2biases1

        self.layer3weights1 = layer3weights1
        self.layer3weights2 = layer3weights2
        self.layer3weights3 = layer3weights3
        self.layer3biases1 = layer3biases1
        self.layer3biases2 = layer3biases2
        self.layer3biases3 = layer3biases3

        self.layer4weights1 = layer4weights1
        self.layer4weights2 = layer4weights2
        self.layer4biases1 = layer4biases1
        self.layer4biases2 = layer4biases2
        return
        
    def softmax(self, x):
        # Subtract the maximum value from each element to avoid overflow
        x = x - np.max(x)
        # Compute the exponentials of each element
        exp_x = np.exp(x)
        # Normalize by dividing each row by the sum of its elements
        return exp_x / np.sum(exp_x)

    def reLU(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # USE neural network to make predictions (forward propagation). Takes in the pixels of an image and creates a prediction of what the digit is.
    def query(self, balance, lives, current_round, towers):

        # Compute the neuron values for layer 1 'l1'
        unactive_l1_top = self.layer1weights1 * np.array([balance, lives, current_round]).reshape(3, 1)
        unactive_l1_top = np.sum(unactive_l1_top, axis=0)
        unactive_l1_top = unactive_l1_top + self.layer1biases1.reshape(16,)
        active_l1_top = self.reLU(unactive_l1_top)

        unactive_l1_bottom = self.layer1weights2 * towers.flatten().reshape(1609632, 1)
        unactive_l1_bottom = np.sum(unactive_l1_bottom, axis=0)
        unactive_l1_bottom = unactive_l1_bottom + self.layer1biases2.reshape(16,)
        active_l1_bottom = self.reLU(unactive_l1_bottom)

        active_l1 = active_l1_top * active_l1_bottom

        unactive_l2 = self.layer2weights1 * active_l1.reshape(16, 1)
        unactive_l2 = np.sum(unactive_l2, axis=0) + self.layer2biases1.reshape(4,)
        active_l2 = self.softmax(unactive_l2)


        if active_l2.argmax() == 0: # If place a tower
            unactive_l3 = active_l2.reshape(4, 1) * self.layer3weights3
            unactive_l3 = np.sum(unactive_l3, axis=0) + self.layer3biases3.reshape(2,)
            active_l3 = self.sigmoid(unactive_l3)

            unactive_l4 = active_l3.reshape(2, 1) * self.layer4weights2
            unactive_l4 = np.sum(unactive_l4, axis=0)
            active_l4 = self.softmax(unactive_l4)

            chosen_tower = active_l4.argmax()
            coordinates = np.array([active_l3[0], active_l3[1]])
            return 'Place Tower', coordinates, chosen_tower

        elif active_l2.argmax() == 1: # If upgrade a tower
            unactive_l3 = active_l2.reshape(4, 1) * self.layer3weights2
            unactive_l3 = np.sum(unactive_l3, axis=0) + self.layer3biases2.reshape(2,)
            active_l3 = self.sigmoid(unactive_l3)

            unactive_l4 = active_l3.reshape(2, 1) * self.layer4weights1
            unactive_l4 = np.sum(unactive_l4, axis=0)
            active_l4 = self.softmax(unactive_l4)
            
            chosen_path = active_l4.argmax()
            coordinates = np.array([active_l3[0], active_l3[1]])
            return 'Upgrade Tower', coordinates, chosen_path

        elif active_l2.argmax() == 2: # If sell a tower
            unactive_l3 = active_l2.reshape(4, 1) * self.layer3weights1
            unactive_l3 = np.sum(unactive_l3, axis=0) + self.layer3biases1.reshape(2,)
            active_l3 = self.sigmoid(unactive_l3)

            coordinates = np.array([active_l3[0], active_l3[1]])
            return 'Sell Tower', coordinates

            
        elif active_l2.argmax() == 3: # If do nothing
            return 'Do Nothing', ''