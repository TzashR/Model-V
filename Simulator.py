# %%
from scipy.stats import gamma

from Generic_Calcs import calculate_point_discrete, make_kernel
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters

hyper_prior_params = (0.5, 0, 1 / 18)

map = create_random_map(20, (2000, 2000), hyper_prior_params)
map.make_neighbors_list_geo()
# reps = generate_random_reporters(1, map.data_points, lambda: 1) #veracity 1
reps = generate_random_reporters(1, map.data_points)  # random veracity
point_calc = calculate_point_discrete
dist_type = gamma

# %%
dist_decay = 0.005
time_decay = 0.4
weight_func = make_kernel(dist_decay, time_decay, 5)

# %%
king = WorldManager(map=map, reporters=reps, point_calc_func=point_calc, dist_type=gamma,
                    prior_params=hyper_prior_params, weight_func=weight_func)
from time import sleep


def dist_tick():
    king.tick()
    king.show_dist('p0')


def dist_loop(days):
    for i in range(days):
        king.tick()
        print(f"day {i}")
        king.show_dist('p0')
        sleep(1.5)


# %%
king.intervention(dist_type(*hyper_prior_params).rvs, king.map.data_points)
king.random_positive_intervention(ratio=0.3)
# %%
king.tick(10, True)
