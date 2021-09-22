# %%
from scipy.stats import gamma

from Generic_Calcs import calculate_point_discrete, make_kernel, add_variance_to_gamma, prediction_loss
from WorldManager import WorldManager, WorldTester
from World_Objects import create_random_map, generate_random_reporters

hyper_prior_params = (0.5, 0, 1 / 18)

map1 = create_random_map(20, (2000, 2000), hyper_prior_params)
map2 = create_random_map(20, (2000, 2000), hyper_prior_params)

map1.make_neighbors_list_geo()
map2.make_neighbors_list_geo()

# reps = generate_random_reporters(1, map.data_points, lambda: 1) #veracity 1
reps1 = generate_random_reporters(5, map1.data_points)  # random veracity
reps2 = generate_random_reporters(5, map2.data_points)  # random veracity

point_calc = calculate_point_discrete
dist_type = gamma

# %%
dist_decay = 0.005
time_decay = 0.4
weight_func1 = make_kernel(dist_decay, time_decay, 5)
weight_func2 = make_kernel(0.004, 0.3, 5)

king1_params = {'map': map1, 'reporters': reps1, 'point_calc_func': point_calc, 'dist_type': gamma,
                'prior_params': hyper_prior_params,
                 'loss_func': prediction_loss,
                'prior_decay_func': lambda prior: add_variance_to_gamma(prior, 0.95)}

# %%
king1 = WorldManager(**king1_params,loss_func=weight_func1)

king2 = WorldManager(map=map2, reporters=reps2, point_calc_func=point_calc, dist_type=gamma,
                     prior_params=hyper_prior_params, weight_func=weight_func2, loss_func=prediction_loss,
                     prior_decay_func=lambda prior: add_variance_to_gamma(prior, 0.95))

from time import sleep


# %%
def dist_tick(world, point_id):
    world.tick()
    world.show_dist(point_id)


def dist_loop(days, world, point_id):
    for i in range(days):
        print(f"day {i}, true s = {world.points_dic[point_id].s}")
        dist_tick(world, point_id)
        sleep(1.5)


# %%
king1.intervention(dist_type(*hyper_prior_params).rvs, king1.map.data_points)
king1.random_positive_intervention(ratio=0.3)
king2.intervention(dist_type(*hyper_prior_params).rvs, king2.map.data_points)
king2.random_positive_intervention(ratio=0.3)

# %%
king1.tick(15, True)


