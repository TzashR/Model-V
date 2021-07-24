from Generic_Calcs import calculate_point_discrete
from World_Objects import DataPoint
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters
from Generic_Calcs import calculate_point_discrete
import numpy as np

import random
import math

def guess_parameters_ground_truth(data_points, neighbors, possible_x, possible_y):
    '''
    A test for the simulation. Supposed to guess the parameters used for calculating points
    :param history: History from world manager
    :param neighbors: A list of neighbors
    :param possible_x: A list of possible coefficents for distance decay
    :return: a tuple(x,y) where x is the estimated distance coefficient and y is the infection factor
    '''

    combinations = [(x, y) for y in possible_y for x in possible_x]
    scores = []
    for c in combinations:
        score = 0
        infection_odds = lambda x: random.uniform(0, 1) < math.exp(-c[0] * x)
        infector_effect = c[1]
        for p in data_points:
            neighbs = neighbors[p.id]
            for t in range(len(p.history)-1):
                p.s = p.history[t][0]
                for n in neighbs:
                    n.s = n.history[t][0]
                pred = calculate_point_discrete(p,neighbs,infection_odds,infector_effect)
                day_score = (p.history[t+1][0]-pred)**2
                score+=day_score
        score = score/(len(data_points)*len(p.history))
        scores.append(score)
        scores = np.array(scores)
        best_index = scores.argmin()
        return combinations[best_index]




map = create_random_map(20,(2000,2000))
map.make_neighbors_list_geo()
reps = generate_random_reporters(8,map.data_points,lambda : 1)
point_calc = calculate_point_discrete

king = WorldManager(map,reps,point_calc)
king.random_intervention(ratio=0.3)

king.tick(30)

py = list(np.arange(0,1,0.01))
px = list(np.arange(0,1,0.001))

res = guess_parameters_ground_truth(king.map.data_points,king.map.neighbors,px,py)
print(res)