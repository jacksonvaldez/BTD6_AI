import numpy as np

layer1weights1 = np.load('trained_params/layer1weights1.npy')
layer1weights2 = np.load('trained_params/layer1weights2.npy')
layer1biases1 = np.load('trained_params/layer1biases1.npy')
layer1biases2 = np.load('trained_params/layer1biases2.npy')

layer2weights1 = np.load('trained_params/layer2weights1.npy')
layer2biases1 = np.load('trained_params/layer2biases1.npy')

layer3weights1 = np.load('trained_params/layer3weights1.npy')
layer3weights2 = np.load('trained_params/layer3weights2.npy')
layer3weights3 = np.load('trained_params/layer3weights3.npy')
layer3biases1 = np.load('trained_params/layer3biases1.npy')
layer3biases2 = np.load('trained_params/layer3biases2.npy')
layer3biases3 = np.load('trained_params/layer3biases3.npy')

layer4weights1 = np.load('trained_params/layer4weights1.npy')
layer4weights2 = np.load('trained_params/layer4weights2.npy')
layer4biases1 = np.load('trained_params/layer4biases1.npy')
layer4biases2 = np.load('trained_params/layer4biases2.npy')

mutation_rate = 0.0000001

layer1weights1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 16))
layer1weights2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (1609632, 16))
layer1biases1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (16, 1))
layer1biases2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (16, 1))

layer2weights1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (16, 4))
layer2biases1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 1))

layer3weights1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 2))
layer3weights2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 2))
layer3weights3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (4, 2))
layer3biases1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (2, 1))
layer3biases2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (2, 1))
layer3biases3_gradient = np.random.uniform(-mutation_rate, mutation_rate, (2, 1))

layer4weights1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (2, 3))
layer4weights2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (2, 23))
layer4biases1_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 1))
layer4biases2_gradient = np.random.uniform(-mutation_rate, mutation_rate, (3, 1))

layer1weights1 = layer1weights1 + layer1weights1_gradient
layer1weights2 = layer1weights2 + layer1weights2_gradient
layer1biases1 = layer1biases1 + layer1biases1_gradient
layer1biases2 = layer1biases2 + layer1biases2_gradient

layer2weights1 = layer2weights1 + layer2weights1_gradient
layer2biases1 = layer2biases1 + layer2biases1_gradient

layer3weights1 = layer3weights1 + layer3weights1_gradient
layer3weights2 = layer3weights2 + layer3weights2_gradient
layer3weights3 = layer3weights3 + layer3weights3_gradient
layer3biases1 = layer3biases1 + layer3biases1_gradient
layer3biases2 = layer3biases2 + layer3biases2_gradient
layer3biases3 = layer3biases3 + layer3biases3_gradient

layer4weights1 = layer4weights1 + layer4weights1_gradient
layer4weights2 = layer4weights2 + layer4weights2_gradient
layer4biases1 = layer4biases1 + layer4biases1_gradient
layer4biases2 = layer4biases2 + layer4biases2_gradient

np.save('mutated_params/layer1weights1', layer1weights1_gradient)
np.save('mutated_params/layer1weights2', layer1weights2_gradient)
np.save('mutated_params/layer1biases1', layer1biases1_gradient)
np.save('mutated_params/layer1biases2', layer1biases2_gradient)

np.save('mutated_params/layer2weights1', layer2weights1_gradient)
np.save('mutated_params/layer2biases1', layer2biases1_gradient)

np.save('mutated_params/layer3weights1', layer3weights1_gradient)
np.save('mutated_params/layer3weights2', layer3weights2_gradient)
np.save('mutated_params/layer3weights3', layer3weights3_gradient)
np.save('mutated_params/layer3biases1', layer3biases1_gradient)
np.save('mutated_params/layer3biases2', layer3biases2_gradient)
np.save('mutated_params/layer3biases3', layer3biases3_gradient)

np.save('mutated_params/layer4weights1', layer4weights1_gradient)
np.save('mutated_params/layer4weights2', layer4weights2_gradient)
np.save('mutated_params/layer4biases1', layer4biases1_gradient)
np.save('mutated_params/layer4biases2', layer4biases2_gradient)