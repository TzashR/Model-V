# %%
from scipy.stats import gamma

from Generic_Calcs import calculate_point_discrete, plot_dist
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters
hyper_prior_params = {'a':0.5,"loc":0, "scale":1/18}

map = create_random_map(1, (2000, 2000),hyper_prior_params )
map.make_neighbors_list_geo()
reps = generate_random_reporters(1, map.data_points, lambda: 1)
point_calc = calculate_point_discrete
dist_type = gamma


# %%
king = WorldManager(map = map, reporters=reps, point_calc_func=point_calc,dist_type = gamma, prior_params=hyper_prior_params)
#%%
king.random_intervention(ratio=1)

#%%
king.tick()
