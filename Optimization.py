import numpy as np
import scipy
from scipy.stats import gamma
from scipy.optimize import  minimize
from Generic_Calcs import calculate_point_discrete, make_kernel
from WorldManager import WorldManager
from World_Objects import create_random_map, generate_random_reporters


def prediction_loss(dist_type: scipy.stats._continuous_distns, dist_params: tuple, true_value, k=100):
    '''
    Calculates a loss of a distribution relating a specific value.
    :param dist_type:
    :param dist_params:
    :param true_value:
    :param k:
    :return:
    '''
    dist = dist_type(*dist_params)
    samples = dist.rvs(size=k)
    probs = dist.pdf(samples)
    loss = sum(probs * (np.square(samples - true_value)))
    return loss


def create_training_set(n_worlds: int, hyper_prior_params: tuple):
    data = []
    weight_func = lambda: 0  # this is irrelevant as we aren't going to predict anything.
    point_calc = calculate_point_discrete

    # loop creating a world each time to extract features and labels from
    for i in range(n_worlds):
        map = create_random_map(20, (2000, 2000), hyper_prior_params)
        map.make_neighbors_list_geo()
        reps = generate_random_reporters(5, map.data_points)  # random veracity
        world = WorldManager(map=map, reporters=reps, point_calc_func=point_calc, dist_type=gamma,
                             prior_params=hyper_prior_params, weight_func=weight_func)
        world.random_positive_intervention(ratio=0.3)
        world.tick(8)
        data += world.extract_data_for_train(5)
    return data


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
    new_dist = gamma.fit(weighted_predictions)
    return new_dist

def get_row_loss(row,weight_func):
    predicted_dist = train_predict(row[0],weight_func)
    return prediction_loss(gamma,predicted_dist,row[1])

def get_dataset_loss(x, dataset):
    '''

    :param x: ndarray shape(2,) containing distance decay and time decay
    :param dataset:
    :return:
    '''
    weight_func = make_kernel(x[0],x[1])
    n = len(dataset)
    loss = 0
    for row in dataset:
        loss+=get_row_loss(row,weight_func)/n
    return loss


#%%
dataset = create_training_set(1,(0.5, 0, 1 / 18))
optim = minimize(get_dataset_loss,np.array((0.005,0.4)),args = dataset)