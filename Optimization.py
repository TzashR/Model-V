import random

import numpy as np
from scipy.stats import gamma

from Generic_Calcs import calculate_point_discrete, make_kernel, add_variance_to_gamma, prediction_loss
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters


# using extract data, so not what i'm actually using
def legacy_create_training_set(n_worlds: int, n_days_per_world: int, hyper_prior_params: tuple, points_per_day=4,
                               all_points=False):
    data = []
    weight_func = lambda: 0  # this is irrelevant as we aren't going to predict anything.
    point_calc = calculate_point_discrete
    loss_func = lambda x: 0  # doesn't matter here

    # loop creating a world each time to extract features and labels from
    for i in range(n_worlds):
        map = create_random_map(20, (2000, 2000), hyper_prior_params)
        map.make_neighbors_list_geo()
        reps = generate_random_reporters(5, map.data_points)  # random veracity
        world = WorldManager(map=map, reporters=reps, point_calc_func=point_calc, dist_type=gamma, loss_func=loss_func,
                             prior_params=hyper_prior_params, weight_func=weight_func,
                             prior_decay_func=lambda prior: add_variance_to_gamma(prior, 0.95))
        world.intervention(gamma(*hyper_prior_params).rvs, world.map.data_points)
        world.random_positive_intervention(ratio=0.3)
        world.tick(n_days_per_world)
        data += world.extract_data_for_train(n_days_per_world, all_points=all_points)
    return data


def create_training_set(n_days: int, hyper_prior_params: tuple, world_params: dict, map_params=(20, (2000, 2000)),
                        n_reporters=5):
    '''
    Creates a dataset, which is essentially the history of reports from world. Also returns the world's map in order to recreate it
    :param n_reporters:
    :param map_parmas:
    :param n_days: How many days the world runs
    :param hyper_prior_params:
    :return: WorldManager.reports object, map and reporters
    '''
    map = create_random_map(*map_params, hyper_prior_params)
    map.make_neighbors_list_geo()
    reps = generate_random_reporters(n_reporters, map.data_points)
    world = WorldManager(map=map, reporters=reps, prior_params=hyper_prior_params, **world_params)
    world.tick(n_days)
    return map, reps, world.reports


def train_predict(X, weight_func, k=128):
    '''
   Function similar to WorldManger.predict_point_neighbors but works with the training data set
    :param params: params to be optimized. In this case time decay and distance decay
    :param X: a tuple of tuples. Each inner tuple is (dist_params,d, t)
    :return: predicted distribution
    '''

    weights = np.array(
        [weight_func(x[1], x[2]) for x in X])
    weights_sum = sum(weights)
    neighbor_predictions = np.column_stack([gamma.rvs(*x[0], size=k) for x in X])
    weighted_predictions = (neighbor_predictions @ weights) / weights_sum
    try:
        new_dist = gamma.fit(weighted_predictions)
    except RuntimeError:
        print('woah')
    return new_dist


def get_row_loss(row, weight_func):
    predicted_dist = train_predict(row[0], weight_func)
    return prediction_loss(gamma, predicted_dist, row[1])


def get_dataset_loss(x, dataset):
    '''

    :param x: ndarray shape(2,) containing distance decay and time decay
    :param dataset:
    :return:
    '''
    assert x[0] > 0 and x[1] > 0
    weight_func = make_kernel(x[0], x[1])
    n = len(dataset)
    loss = 0
    for row in dataset:
        loss += get_row_loss(row, weight_func) / n
    return loss


def brute_force_optim_world(possible_dist_decay, possible_time_decay, n_worlds):
    combinations = [(x, y) for y in possible_dist_decay for x in possible_time_decay]
    losses = []
    true_params = random.choice(combinations)
    hyper_prior_params = (0.5, 0, 1 / 18)
    map = create_random_map(20, (2000, 2000), hyper_prior_params)
    map.make_neighbors_list_geo()
    reps = generate_random_reporters(5, map.data_points)  # random veracity
    weight_func = lambda: 0  # this is irrelevant as we aren't going to predict anything.
    point_calc = calculate_point_discrete
    best_c = None
    best_score = float('inf')

    for c in combinations:
        c_score = 0
        for i in range(n_worlds):
            world_score = 0
            world = WorldManager(map=map, reporters=reps, point_calc_func=point_calc, dist_type=gamma,
                                 prior_params=hyper_prior_params, weight_func=weight_func, loss_func=prediction_loss,
                                 prior_decay_func=lambda prior: add_variance_to_gamma(prior, 0.95))
            world.intervention(gamma(*hyper_prior_params).rvs, world.map.data_points)
            world.random_positive_intervention(ratio=0.3)
            world.tick(30)
            for day in range(335):
                world.tick()
                world_score += world.current_average_loss() / 335
            c_score += world_score / n_worlds
        losses.append((c, c_score))
        if c_score < best_score:
            best_c = c
            best_score = c_score
        if c == true_params:
            true_params_score = c_score
    print(f"true_params = {true_params}, score {true_params_score} "
          f"best params = {best_c}, score {best_score}")
    return losses


def brute_force_optim(possible_dist_decay, possible_time_decay, dataset):
    combinations = [(x, y) for y in possible_dist_decay for x in possible_time_decay]
    best_loss = float('inf')
    best_c = None
    for c in combinations:
        try:
            loss = get_dataset_loss(c, dataset)
        except:
            continue
        if loss < best_loss:
            best_loss = loss
            best_c = c
    return best_c, best_loss


# %%
dataset = legacy_create_training_set(1, 400, (0.5, 0, 1 / 18))
# %%
possible_dist_decay = list(np.linspace(0.0001, 1, 5))
possible_time_decay = list(np.linspace(0.00001, 5, 5))

# %%
brute_force_optim_world(possible_dist_decay, possible_time_decay, 2)
# %%
res = brute_force_optim(possible_dist_decay, possible_time_decay, dataset)
# %%
# bounds = Bounds((0.0001,0.0001),(3,3))
# optim = minimize(get_dataset_loss,np.array((0.005,1)),bounds = bounds, args = dataset)
# print(optim)
