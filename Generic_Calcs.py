'''
Functions used on data points. Useful when generating data,training the model and when predicting
'''

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats._continuous_distns
from scipy.special import expit


# True calculations - what the simulations uses, not what "we" know


def calculate_point_discrete(target, neighbors,
                             infection_odds=lambda x: random.uniform(0, 1) < math.exp(-0.005 * x),
                             infector_effect=0.03):
    '''Calculates the target's value based on the set of sources using discrete function
        This is a simplification and we use it to start off
    :param target: Datapoint to get the value
    :param neighbors: Datapoint's neighbors
    :param infection_odds:A function that returns true if the neighbor infects the target
    :return: The new value of the datapoint's s.
    '''
    infectors = 0
    if target.s > 0:  # if target is infected, things can get worse even without external infection
        infectors += 1
    for neighbor in neighbors:
        if neighbor.s > 0:  # only infected neighbors can infect
            dist = target.calc_distance(neighbor)
            if infection_odds(dist):
                infectors += 1
    infection_factor = infector_effect * infectors
    new_s = min(target.s + infection_factor, 1)
    return new_s


def fit_average_posterior(dist1_params: tuple, dist2_params: tuple, dist1_type, dist2_type, weights, result_dist_type,
                          n_samples=10000):
    '''
    Takes two distributions, samples from both, for each observation calculates weighted avreage according to weights
    and fits a new distribution
    :param dist1_params: Parameters for first distribution. Should be a tuple
    :param dist2_params: Same as dist1
    :param dist1_type: type of dist1. Should have rvs method
    :param dist2_type: type of dist2. Should have rvs method
    :param weights: a tuple stating how much weight to give each distribution
    :param result_dist_type: A type of distribution e.g. gamma
    :param n_samples:  how many samnples to take from each distribution
    :return: Distribution params, using the result_dist_type fit method.
    '''
    assert (sum(weights) == 1), f"weights should add up to 1. Provided weights are {weights}"

    dist1_samples = dist1_type.rvs(*dist1_params, size=n_samples)  # returns a vector of n_samples samples
    dist2_samples = dist2_type.rvs(*dist2_params, size=n_samples)

    weighted_samples = weights[0] * dist1_samples + weights[1] * dist2_samples
    res = result_dist_type.fit(weighted_samples)
    return res


def make_kernel(alpha, beta, time_limit=5):
    '''
    Creates the function that will be used as the kernel for points effecting each other
    :param alpha: Decay in distance parameter
    :param beta: Decay in time parameter
    :param time_limit: Max days after which weight is 0
    :return: Function  that takes in observation and a point and calculates observations weight in affecting the point
    '''

    def calc_weight(distance: float, time_diff: int, dist_decay=alpha, time_decay=beta):
        '''
        :param distance: distance between the points
        :param time_diff: time difference current_time - report_time
        :param dist_decay:  parameter for decaying over distance
        :param time_decay:  parameter for decaying over time
        :return: source's weight in predicting target's s
        '''
        assert time_diff >= 0
        if time_diff >= time_limit: return 0
        try:
            weight = math.exp(-time_decay * time_diff - dist_decay * distance)
        except OverflowError:
            print('woosh')
            print(weight)
        return weight

    return calc_weight


def calc_adj_mat(points):
    '''
    Calculates adjacency matrix
    :param points: ndarray of points
    :return: ndarray that is the adjacency matrix
    '''
    n = len(points)
    res_mat = np.zeros(shape=(n, n))
    for i in range(n):
        # the weight formula calculated on the entire row:
        res_mat[i] = np.linalg.norm(points - points[i], axis=1)
    np.fill_diagonal(res_mat, 0)
    return res_mat


def plot_dist(dist_type: scipy.stats._continuous_distns, dist_params: tuple, annotate_value=None):
    '''
    Plots a graph of the distribution
    :param dist_type: distribution type with rvs function
    :param dist_params: params for the distribution
    :return:
    '''
    x = np.linspace(0, 1, 1000)

    dist = dist_type(*dist_params)
    y = dist.pdf(x)
    plt.plot(x, y)

    if annotate_value is not None:
        plt.scatter([annotate_value], [0], color='red')

    if dist_type == scipy.stats.norm:
        mu, sigma = dist_params
        plt.title(
            f'mu = {round(mu, 2)}, sigma = {round(sigma, 2)}')

    else:
        a, loc, scale = dist_params
        plt.title(
            f'a = {round(a, 2)}, scale = {round(scale, 2)}, loc = {round(loc, 2)}, mean = {round(a * scale, 2)}, SD = {round(a * (scale ** 2), 2)}')
    plt.show()


def get_range_from_dist(dist_type: scipy.stats._continuous_distns, dist_params: tuple, percentiles=(30, 70)) -> tuple:
    '''
    :param dist_type: a scipy dist type (must have rvs method), e.g gamma
    :param dist_params: tuple with the dist params
    :param percentiles: desired percentiles to be calculated
    :return: a tuple with lower bound and upper bound, the bounds a the percentiles provided
    '''
    return tuple(np.percentile(dist_type.rvs(*dist_params, 1000), percentiles))


def add_variance_to_gamma(dist_params: tuple, factor: float) -> tuple:
    return dist_params[0] * factor, dist_params[1], dist_params[2] * (1 / factor)


def add_variance_to_norm(dist_params: tuple, factor: float) -> tuple:
    return dist_params[0], dist_params[1] * factor


def prediction_loss(dist_type: scipy.stats._continuous_distns, dist_params: tuple, true_value, k=1000,
                    epsilon_fix=0.01):
    '''
    Calculates a loss of a distribution relating a specific value.
    :param dist_type:
    :param dist_params:
    :param true_value:
    :param k:
    :return:
    '''
    dist = dist_type(*dist_params)
    fixed_number = dist_params[1] + epsilon_fix
    samples = dist.rvs(size=k)
    samples[samples < fixed_number] = fixed_number
    true_value = max(fixed_number, true_value)
    probs = dist.pdf(samples)
    loss = sum((1 / k) * probs * (np.square(samples - true_value)))
    assert loss < float('inf')
    return loss


def agg_noraml_dists(dists_params):
    '''
    Given a list of params of normal distribution ((mu_1,sigma_1),...,(mu_n,sigma_n)),
     returns a new distribution with mu = mu_1/sigma_1 + ... mu_n/sigma_n
     and sigma = (sigma_1 +...+sigma_n)/(sigma_1*...*sigma_n)
    :param dists_params: List of distributions
    :return: mu,sigma of the new distribution
    '''
    new_mu = sum([dist[0] / dist[1] for dist in dists_params])
    sigmas = [dist[1] for dist in dists_params]
    new_sigma = sum(sigmas) / math.prod(sigmas)

    return (new_mu, new_sigma)


def prediction_unit_func_normal(dists_params, sds_gap=1):
    mu, sigma = agg_noraml_dists(dists_params)
    prediction = expit([mu - (sds_gap * sigma), mu + (sds_gap * sigma)])
    return prediction


def  likelihood_alphas(alphas: np.array, features: np.array, af_obs, is_obs, return_minus: bool):
    '''
    Calculates the likelihood of seeing the distances (is_obs
    :param is_obs:
    :param af_obs:
    :param alphas:np.array Weight of every feature, dimension d
    :param features: features of every local sample, dimensions n,d
    :return:
    '''
    distances = np.square(is_obs - af_obs)
    first_term = -0.5 * sum(features @ alphas)
    second_term = -(distances / 2) @ np.exp(-((features @ alphas)))
    res = first_term + second_term
    if return_minus: res = -res
    return res


def predict_predicton_unit(X, alphas, obs):
    '''
    Calculates the most likely s value for the unit
    :param X:
    :param alphas:
    :param obs:
    :return:
    '''
    sigmas = X @ alphas
    weights = 1/sigmas
    res_mu = sum(obs / (np.sqrt(sigmas)))
    res_sigma = sum(sigmas) / (np.product(np.sqrt(sigmas)))
    return res_mu, res_sigma


def likelihood_fixed_sigma(sigma_square: float,is_obs:np.array,af_obs:np.array,return_minus:bool):
    '''
    Sums the likelihood of all observations assuming normal dist where is_sample is the mean
    and sigma is the SD
    :param sigma:
    :param distances: squared distances between african samples and Israeli samples
    :return:
    '''
    distances = np.square(is_obs-af_obs)
    res = sum(-0.5*np.log(np.exp((-1/(2*sigma_square))*distances)))
    if return_minus: return -res

    return res


def generate_none_spatial_data(n_features, n_samples,min_obs_village = 1, max_obs_village=10, weights=None, split_train_test=False,
                               ratio=0.2,max_sigma = 100):
    '''
    Generates data similar to the water reports. We can use this to test our model where the goal is to learn
    the weights from the features and obs returned
    :param n_features:
    :param n_samples:
    :param weights:
    :return:
    '''

    if weights is None:
        ## I will generate weights s.t their sum is 4.6 because features are 0-1 and I want max sigma to be 100.
        ## so 4.6 is ln(100)
        weight_sum = math.log(max_sigma**2)
        weights = np.random.uniform(0, 100, n_features)
        weights = weights/(sum(weights))*weight_sum
    assert n_features == len(weights)

    villages_division = []

    village_num = 0
    while len(villages_division) < n_samples:
        n_obs_in_village = random.randint(min_obs_village, max_obs_village)
        villages_division += [village_num] * n_obs_in_village
        village_num += 1

    villages_division = np.array(villages_division[:n_samples])
    villages = np.unique(villages_division)
    features = np.random.rand(n_samples, n_features)  # array n_samples X n_features, each feature ~U(0,1)
    variances = np.exp(features @ weights)
    israeli_obs = np.random.randint(0, 100,n_samples) * 0.6+20
    obs_no_clip = np.random.normal(israeli_obs, np.sqrt(variances))
    obs = np.clip(obs_no_clip, 0, 100)

    count_clip_diff = sum(obs_no_clip != obs)
    # print(f"count_clip_diff = {count_clip_diff}")

    if not split_train_test:
        return {'features': features, 'israeli_obs': israeli_obs, 'af_obs': obs, 'weights': weights,
                'villages_division': villages_division, 'villages': villages}

    n_train_villages = round((1 - ratio) * n_samples)
    train_villages = list(villages_division[:n_train_villages])
    test_villages = list(villages_division[n_train_villages:])

    assert (len(train_villages) + len(test_villages) == n_samples)

    while train_villages[-1] == test_villages[0]:  # not to have same village in different sets
        test_villages.insert(0, train_villages.pop(-1))

    train_villages = np.array(train_villages)
    test_villages = np.array(test_villages)
    features_train = features[:len(train_villages)]
    features_test = features[len(train_villages):]
    variances_train = variances[:len(train_villages)]
    variances_test = variances[len(train_villages):]
    is_obs_train = israeli_obs[:len(train_villages)]
    is_obs_test = israeli_obs[len(train_villages):]
    obs_train = obs[:len(train_villages)]
    obs_test = obs[len(train_villages):]


    return {'f_train': features_train, 'f_test': features_test, 'variances_train': variances_train,
            'variances_test': variances_test, 'is_train': is_obs_train, 'is_test': is_obs_test, 'obs_train': obs_train,
            'obs_test': obs_test, 'test_villages': test_villages, "train_villages": train_villages,
            "villages": villages,'weights':weights}

def predict_variances(X,alphas):
    return  np.sqrt(np.exp(X@alphas))