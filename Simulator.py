from WorldManager import WorldManager
from World_Objects import create_random_map
from Generic_Calcs import calculate_point_discrete
import numpy as np
import random

#%%
map = create_random_map(20,(2000,2000))
king = WorldManager(map,)