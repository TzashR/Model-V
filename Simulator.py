#%%
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters
from Generic_Calcs import calculate_point_discrete
import numpy as np
import random

map = create_random_map(20,(2000,2000))
map.make_neighbors_list_geo()
reps = generate_random_reporters(8,map.data_points,lambda : 1)
point_calc = calculate_point_discrete

#%%
king = WorldManager(map,reps,point_calc)
king.random_intervention(ratio=0.3)
