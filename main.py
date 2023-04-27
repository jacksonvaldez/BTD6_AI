import pyautogui # Allows for program to control mouse movements
import time
import numpy as np
import pdb
from neural_network import NeuralNetwork
import math

params_path = 'mutated'

layer1weights1 = np.load(f"{params_path}_params/layer1weights1.npy")
layer1weights2 = np.load(f"{params_path}_params/layer1weights2.npy")
layer1biases1 = np.load(f"{params_path}_params/layer1biases1.npy")
layer1biases2 = np.load(f"{params_path}_params/layer1biases2.npy")

layer2weights1 = np.load(f"{params_path}_params/layer2weights1.npy")
layer2biases1 = np.load(f"{params_path}_params/layer2biases1.npy")

layer3weights1 = np.load(f"{params_path}_params/layer3weights1.npy")
layer3weights2 = np.load(f"{params_path}_params/layer3weights2.npy")
layer3weights3 = np.load(f"{params_path}_params/layer3weights3.npy")
layer3biases1 = np.load(f"{params_path}_params/layer3biases1.npy")
layer3biases2 = np.load(f"{params_path}_params/layer3biases2.npy")
layer3biases3 = np.load(f"{params_path}_params/layer3biases3.npy")

layer4weights1 = np.load(f"{params_path}_params/layer4weights1.npy")
layer4weights2 = np.load(f"{params_path}_params/layer4weights2.npy")
layer4biases1 = np.load(f"{params_path}_params/layer4biases1.npy")
layer4biases2 = np.load(f"{params_path}_params/layer4biases2.npy")

neural_net = NeuralNetwork(layer1weights1, layer1weights2, layer1biases1, layer1biases2, layer2weights1, layer2biases1, layer3weights1, layer3weights2, layer3weights3, layer3biases1, layer3biases2, layer3biases3, layer4weights1, layer4weights2, layer4biases1, layer4biases2)

towers = np.load('game_state/towers.npy')
balance = input("Please enter user balance: ")
balance = int(balance)
lives = input("Please enter user lives: ")
lives = int(lives)
current_round = input("Please enter the current round: ")
current_round = int(current_round)

# pdb.set_trace()


query = neural_net.query(balance, lives, current_round, towers)


def distance(point_1, point_2):
	return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def closest_tower_coordinates(coordinates, towers):
	coordinates = coordinates * [162, 108]

	tower_args = np.argwhere(towers == 1) # Returns a list of arguments where the element is equal to 1
	tower_coordinates = np.unique(tower_args[:, 0:2], axis=0)
	distances = np.array([])
	for x in tower_coordinates:
		distances = np.append(distances, distance(x, coordinates))
	if np.size(distances) == 0:
		return False
	chosen_tower = tower_coordinates[distances.argmin()]

	return chosen_tower







if query[0] == 'Place Tower':
	pyautogui.click(x=48, y=0)
	pyautogui.click(x=48, y=0)

	tower_names = ['Dart Monkey', 'Boomerang Monkey', 'Bomb Shooter', 'Tack Shooter', 'Ice Monkey', 'Glue Gunner', 'Sniper Monkey', 'Monkey Sub', 'Monkey Buccaneer', 'Monkey Ace', 'Heli Pilot', 'Mortar Monkey', 'Dartling Gunner', 'Wizard Monkey', 'Super Monkey', 'Ninja Monkey', 'Alchemist', 'Druid', 'Banana Farm', 'Spike Factory', 'Monkey Village', 'Engineer Monkey', 'Beast Handler']
	coordinates = [(query[1][0] * 1620).round(-1) + 24, (query[1][1] * 1080).round(-1)]
	chosen_tower = tower_names[query[2]]
	print(f"Place a {chosen_tower} at coordinates {coordinates[0]} {coordinates[1]}")

	if query[2] <= 10:
		pyautogui.click(x=1770, y=950)
		pyautogui.scroll(50)
	else:
		pyautogui.click(x=1770, y=950)
		pyautogui.scroll(-50)
	time.sleep(0.5)

	x_positions = [1710, 1830]
	y_positions = [210, 340, 475, 610, 740, 880]

	position = int(query[2] + 1)
	x_position = x_positions[position % 2]
	y_position = y_positions[(position % 12) // 2] # The // division sign is the same as regular division but integer division
	
	pyautogui.click(x=x_position, y=y_position) # Select Tower
	pyautogui.click(x=coordinates[0], y=coordinates[1]) # Place Tower

	success = input("Did the tower place successfuly? (y/n) ")
	if success == 'y':
		position = [(query[1][0] * 162).round(), (query[1][1] * 108).round()]
		towers[int(position[0]), int(position[1]), query[2] * 4] = 1
		np.save('game_state/towers.npy', towers)


elif query[0] == 'Upgrade Tower':
	pyautogui.click(x=48, y=0)
	pyautogui.click(x=48, y=0)

	path_names = ['Top Path', 'Middle Path', 'Bottom Path']
	coordinates = [(query[1][0] * 1620).round(2) + 24, (query[1][1] * 1080).round(2)]
	chosen_path = path_names[query[2]]
	print(f"Upgrade the {chosen_path} at tower closest to coordinates {coordinates[0]} {coordinates[1]}")

	chosen_tower_location = closest_tower_coordinates(query[1], towers)

	if type(chosen_tower_location) == np.ndarray:
		click_coordinates = [(chosen_tower_location[0] * 10).round(2) + 24, (chosen_tower_location[1] * 10).round(2)]
		print(f" ----> Tower At {click_coordinates[0]} {click_coordinates[1]}")

		pyautogui.click(x=click_coordinates[0], y=click_coordinates[1])

		if click_coordinates[0] >= 835:
			print("Tower is on the right")
			upgrade_clicks = [[330, 490], [330, 640], [330, 790]] # [top, middle, bottom]
			upgrade_click = upgrade_clicks[query[2]]
			time.sleep(0.1)
			pyautogui.click(x=upgrade_click[0], y=upgrade_click[1])

		elif click_coordinates[0] < 835:
			print("Tower is on the left")
			upgrade_clicks = [[1550, 490], [1550, 640], [1550, 790]] # [top, middle, bottom]
			upgrade_click = upgrade_clicks[query[2]]
			time.sleep(0.1)
			pyautogui.click(x=upgrade_click[0], y=upgrade_click[1])


	success = input("Did the tower upgrade successfuly? (y/n) ")
	if success == 'y':
		argus = np.argwhere(towers[chosen_tower_location[0], chosen_tower_location[1]] == 1)
		argus = argus[argus % 4 == 0]
		argus = argus + (query[2] + 1)

		towers[chosen_tower_location[0], chosen_tower_location[1], argus] += 1
		np.save('game_state/towers.npy', towers)


elif query[0] == 'Sell Tower':
	pyautogui.click(x=48, y=0)
	pyautogui.click(x=48, y=0)

	coordinates = [(query[1][0] * 1620).round(2) + 24, (query[1][1] * 1080).round(2)]
	print(f"Sell tower closest to coordinates {coordinates[0]} {coordinates[1]}")

	chosen_tower_location = closest_tower_coordinates(query[1], towers)

	if type(chosen_tower_location) == np.ndarray:
		click_coordinates = [(chosen_tower_location[0] * 10).round(2) + 24, (chosen_tower_location[1] * 10).round(2)]
		print(f" ----> Tower At {click_coordinates[0]} {click_coordinates[1]}")

		pyautogui.click(x=click_coordinates[0], y=click_coordinates[1])

		if click_coordinates[0] >= 835:
			print("Tower is on the right")
			time.sleep(0.1)
			pyautogui.click(x=325, y=910)

		elif click_coordinates[0] < 835:
			print("Tower is on the left")
			time.sleep(0.1)
			pyautogui.click(x=1545, y=910)


	success = input("Did the tower sell successfuly? (y/n) ")
	if success == 'y':
		towers[chosen_tower_location[0], chosen_tower_location[1]] = 0
		np.save('game_state/towers.npy', towers)


elif query[0] == 'Do Nothing':
	print("Do Nothing")