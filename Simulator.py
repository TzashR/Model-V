# %%
from scipy.stats import gamma

from Generic_Calcs import calculate_point_discrete
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters

map = create_random_map(20, (2000, 2000))
map.make_neighbors_list_geo()
reps = generate_random_reporters(8, map.data_points, lambda: 1)
point_calc = calculate_point_discrete

def gamma_dist(alpha, beta) :
    return gamma.rvs(a=alpha, scale=beta)

# %%
king = WorldManager(map, reps, point_calc,dist_type,(0.5,0.2))
king.random_intervention(ratio=0.3)
