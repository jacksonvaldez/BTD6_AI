import numpy as np

layer1weights1 = np.random.uniform(-0.5, 0.5, (3, 16))
layer1weights2 = np.random.uniform(-0.5, 0.5, (1609632, 16))
layer1biases1 = np.full(16, 0, dtype=np.float64).reshape(16, 1)
layer1biases2 = np.full(16, 0, dtype=np.float64).reshape(16, 1)

layer2weights1 = np.random.uniform(-0.5, 0.5, (16, 4))
layer2biases1 = np.full(4, 0, dtype=np.float64).reshape(4, 1)

layer3weights1 = np.random.uniform(-0.5, 0.5, (4, 2))
layer3weights2 = np.random.uniform(-0.5, 0.5, (4, 2))
layer3weights3 = np.random.uniform(-0.5, 0.5, (4, 2))
layer3biases1 = np.full(2, 0, dtype=np.float64).reshape(2, 1)
layer3biases2 = np.full(2, 0, dtype=np.float64).reshape(2, 1)
layer3biases3 = np.full(2, 0, dtype=np.float64).reshape(2, 1)

layer4weights1 = np.random.uniform(-0.5, 0.5, (2, 3))
layer4weights2 = np.random.uniform(-0.5, 0.5, (2, 23))
layer4biases1 = np.full(3, 0, dtype=np.float64).reshape(3, 1)
layer4biases2 = np.full(23, 0, dtype=np.float64).reshape(23, 1)


np.save('trained_params/layer1weights1', layer1weights1)
np.save('trained_params/layer1weights2', layer1weights2)
np.save('trained_params/layer1biases1', layer1biases1)
np.save('trained_params/layer1biases2', layer1biases2)

np.save('trained_params/layer2weights1', layer2weights1)
np.save('trained_params/layer2biases1', layer2biases1)

np.save('trained_params/layer3weights1', layer3weights1)
np.save('trained_params/layer3weights2', layer3weights2)
np.save('trained_params/layer3weights3', layer3weights3)
np.save('trained_params/layer3biases1', layer3biases1)
np.save('trained_params/layer3biases2', layer3biases2)
np.save('trained_params/layer3biases3', layer3biases3)

np.save('trained_params/layer4weights1', layer4weights1)
np.save('trained_params/layer4weights2', layer4weights2)
np.save('trained_params/layer4biases1', layer4biases1)
np.save('trained_params/layer4biases2', layer4biases2)
