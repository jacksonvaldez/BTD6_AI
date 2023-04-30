# Initialize a starting game state (full lives, starting balance, starting round, no towers)
import numpy as np

towers = np.full(1609632, 0, dtype=np.float64).reshape(162, 108, 92)

np.save('game_state/towers.npy', towers)
