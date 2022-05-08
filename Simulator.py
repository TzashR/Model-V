# %%
from scipy.stats import gamma, norm

from Generic_Calcs import calculate_point_discrete, make_kernel, add_variance_to_gamma,add_variance_to_norm, prediction_loss, prediction_unit_func_normal
from WorldManager import WorldManager, WorldTester
from World_Objects import create_random_map, generate_random_reporters
from time import sleep

hyper_prior_params = (-5,1)

map = create_random_map(20, (2000, 2000), hyper_prior_params)

map.make_neighbors_list_geo()

# reps = generate_random_reporters(1, map.data_points, lambda: 1) #veracity 1
reps = generate_random_reporters(5, map.data_points)  # random veracity

point_calc = calculate_point_discrete
dist_type = norm

 # %%
dist_decay = 0.005
time_decay = 0.4
weight_func = make_kernel(dist_decay, time_decay, 5)

king_params = {'map': map, 'reporters': reps, 'point_calc_func': point_calc, 'dist_type': dist_type,
                'prior_params': hyper_prior_params,
                 'loss_func': prediction_loss,
                'prior_decay_func': lambda prior: add_variance_to_norm(prior, 1.1), 'prediction_unit_func' : prediction_unit_func_normal}

# %%
king = WorldManager(**king_params,weight_func=weight_func)

for point in map.data_points:
    king.add_prediction_unit([point],point.id)



# %%
def dist_tick(world, point_id):
    world.tick()
    world.show_dist(point_id)


def dist_loop(days, world, point_id):
    for i in range(days):
        print(f"day {i}, true s = {world.points_dic[point_id].s}, point loss = {world.point_loss(point_id)}")
        dist_tick(world, point_id)
        sleep(1.5)


# %%
# king2.intervention(dist_type(*hyper_prior_params).rvs, king2.map.data_points)
king.random_positive_intervention(ratio=0.3)

# %%
king.tick(15, True)


