# Initialize a starting game state (full lives, starting balance, starting round, no towers)
import numpy as np

np.save('game_state/balance.npy', np.array([0.0650]) ) # balance / 10000
np.save('game_state/current_round.npy', np.array([0.0375]) ) # current_round / 80
np.save('game_state/lives.npy', np.array([1.0]) ) # lives / 100

towers = np.full(1609632, 0, dtype=np.float64).reshape(162, 108, 92)

np.save('game_state/towers.npy', towers)
