'''
Functions used on data points. Useful when generating data,training the model and when predicting
'''

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats._continuous_distns
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


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


def likelihood_alphas(alphas: np.array, features: np.array, reported_y, true_y, return_minus=True):
    '''
    Calculates the likelihood of seeing the distances (is_obs
    :param true_y:
    :param reported_y:
    :param alphas:np.array Weight of every feature, dimension d
    :param features: features of every local sample, dimensions n,d
    :return:
    '''
    distances = np.square(true_y - reported_y)
    variances = generate_variances(features, alphas)
    res = -0.5 * (sum(distances / variances + np.log(variances)))
    if return_minus:
        res = -res
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
    weights = 1 / sigmas
    res_mu = sum(obs / (np.sqrt(sigmas)))
    res_sigma = sum(sigmas) / (np.product(np.sqrt(sigmas)))
    return res_mu, res_sigma


def likelihood_fixed_sigma(sigma_square: float, is_obs: np.array, af_obs: np.array, return_minus: bool):
    '''
    Sums the likelihood of all observations assuming normal dist where is_sample is the mean
    and sigma is the SD
    :param sigma:
    :param distances: squared distances between african samples and Israeli samples
    :return:
    '''
    distances = np.square(is_obs - af_obs)
    res = sum(-0.5 * np.log(np.exp((-1 / (2 * sigma_square)) * distances)))
    if return_minus: return -res

    return res


def generate_none_spatial_data(n_features, n_samples, min_obs_village=1, max_obs_village=10, feature_weights=None,
                               split_train_test=False,
                               train_test_ratio=0.3, max_sigma=100, X=None, villages_divison=None, n_cat_features=None,
                               feature_raise_power=1, beta_feautre_col=True):
    '''
    Generates data similar to the water reports. We can use this to test our model where the goal is to learn
    the weights from the features and obs returned
    :param feature_raise_power: The power to wich we raise the feautres.
    This is here is a cheap trick to somewhat control the feature's distribution.
     In some of the experiments I want them to be rather high so I take their root (power = 1/3 for example)
    :param train_test_ratio:
    :param n_features:
    :param n_samples:
    :param feature_weights:
    :return:
    '''

    if feature_weights is None:
        ## W ill generate weights s.t their sum is 4.6 because features are 0-1 and I want max sigma to be 100.
        ## so 4.6 is ln(100)
        feature_weights = np.random.uniform(-10, 10,
                                            n_features + 1 if beta_feautre_col else n_features)  # TODO how big can they get?

        ### I normalize the positives and the negatives separately
        # feature_weights = feature_weights/(sum(feature_weights))*2*math.log(max_sigma) #TODO do I need to normalize this?
        positives = feature_weights[feature_weights > 0]
        negatives = feature_weights[feature_weights < 0]

        ##Talk about this normalization
        positives = positives / (sum(positives)) * 2 * math.log(max_sigma)
        negatives = -1 * negatives / (sum(negatives)) * math.log(max_sigma)

        feature_weights[feature_weights > 0] = positives
        feature_weights[feature_weights < 0] = negatives

    if beta_feautre_col:
        assert(n_features + 1 == len(feature_weights))
    else:
        assert ( n_features == len(feature_weights))

    if villages_divison is None:
        villages_division = []
        actual_y = []

        village_num = 0
        while len(villages_division) < n_samples:
            n_obs_in_village = random.randint(min_obs_village, max_obs_village)
            village_y = random.randint(0,100)

            villages_division += [village_num] * n_obs_in_village
            actual_y += [village_y]*n_obs_in_village

            village_num += 1

        villages_division = np.array(villages_division[:n_samples])
        actual_y = np.array(actual_y[:n_samples])

        assert len(actual_y) == len(villages_division)

    villages = np.unique(villages_division)

    assert (np.unique(villages_division, return_counts=True)[1] >= min_obs_village).all() & (
            np.unique(villages_division, return_counts=True)[
                1] <= max_obs_village).all(), "Obs per village not what you wanted"

    if n_cat_features is None:
        n_cat_features = random.randint(0, n_features)
    n_con_features = n_features - n_cat_features

    con_features = np.random.rand(n_samples, n_con_features) ** (
        feature_raise_power)  # array n_samples X n_con_features, each feature ~U(0,1)
    cat_features = np.random.choice([0, 1], (n_samples, n_cat_features))

    if X is None:
        X = np.concatenate([con_features, cat_features], axis=1)
        if beta_feautre_col:
            X = np.concatenate([X, np.ones(shape=(n_samples, 1))],axis = 1)

    variances = generate_variances(X, feature_weights)


    obs_no_clip = np.random.normal(actual_y, np.sqrt(variances))
    obs = np.clip(obs_no_clip, 0, 100)

    count_clip_diff = sum(obs_no_clip != obs)
    # print(f"count_clip_diff = {count_clip_diff}")

    df = pd.DataFrame({'true_y': actual_y, 'local_y': obs, "variance": variances})
    df = pd.concat([df, pd.DataFrame(X)], axis=1)

    if not split_train_test:
        return {'X': X, 'israeli_obs': actual_y, 'af_obs': obs, 'weights': feature_weights,
                'villages_division': villages_division, 'villages': villages, "df": df}

    n_train_villages = round((1 - train_test_ratio) * len(villages))
    train_villages = villages[:n_train_villages]
    test_villages = villages[n_train_villages:]

    train_indices = np.isin(villages_division, train_villages)
    test_indices = ~train_indices

    df['is_train'] = train_indices

    assert (sum(train_indices) + sum(test_indices) == n_samples)

    X_train = X[train_indices]
    X_test = X[test_indices]
    variances_train = variances[train_indices]
    variances_test = variances[test_indices]
    is_obs_train = actual_y[train_indices]
    is_obs_test = actual_y[test_indices]
    obs_train = obs[train_indices]
    obs_test = obs[test_indices]

    return {'X_train': X_train, 'X_test': X_test, 'variances_train': variances_train,
            'variances_test': variances_test, 'is_train': is_obs_train, 'is_test': is_obs_test, 'obs_train': obs_train,
            'obs_test': obs_test, 'test_villages': test_villages, "train_villages": train_villages,
            "villages": villages, 'weights': feature_weights,"df":df,'villages_division': villages_division}


def generate_variances(X, alphas, noise=None):
    if noise is None:
        noise = 0
    else:
        assert (len(noise) == X.shape[0])
    return np.exp(X @ alphas + noise)


def water_model_loss(bias, X, villages_division, obs, is_obs, alphas, for_optim=False):
    villages_unique = np.unique(villages_division)
    variances = generate_variances(X, alphas, bias)

    res_dic = {}

    true_y = []
    naive_y = []
    model_y = []

    for v in villages_unique:
        indices = (villages_division == v)
        variances_v = variances[indices]

        # print(f" indices = {indices},v = {v}")
        obs_v = obs[indices]
        is_obs_v = is_obs[indices]
        true_av = is_obs_v.mean()
        true_y.append(true_av)

        # model_weights = np.reciprocal(variances_v) + bias
        model_weights = np.reciprocal(variances_v)
        model_av = np.dot(model_weights, obs_v) / sum(model_weights)
        model_y.append(model_av)

        naive_av = obs_v.mean()
        naive_y.append(naive_av)

    true_y = np.array(true_y)
    model_y = np.array(model_y)
    naive_y = np.array(naive_y)

    naive_loss_reg = mean_squared_error(naive_y, true_y)
    model_loss_reg = mean_squared_error(model_y, true_y)

    res_dic['reg'] = (naive_loss_reg, model_loss_reg)

    if for_optim:
        return model_loss_reg
    return res_dic


def loss_per_agg_level(data_df, obs_df, agg_size: int, measured_val: str, train_size=0.7):
    n_samples = len(data_df)
    n_villages = n_samples // agg_size

    village_ids = list(range(n_villages + 1))
    villages = agg_size * list(range(n_villages)) + [n_villages + 1] * (n_samples % agg_size)
    random.shuffle(villages)
    data_df['village_id'] = villages

    train_village_ids = set(random.sample(village_ids, math.floor(train_size * len(village_ids))))
    test_village_ids = set(village_ids).difference(train_village_ids)

    train_indices = data_df['village_id'].isin(train_village_ids)
    test_indices = ~train_indices

    train_villages = df[train_indices]['village_id']
    test_villages = df[test_indices]['village_id']

    data_df = data_df.drop(columns=['village_id'])

    X_train = data_df[train_indices]
    X_test = data_df[test_indices]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = obs_df[train_indices]
    is_obs_train = y_train[measured_val + "_y"]
    af_obs_train = y_train[measured_val + "_x"]
    y_test = obs_df[test_indices]

    is_obs_test = y_test[measured_val + "_y"]
    af_obs_test = y_test[measured_val + "_x"]

    alphas_0 = np.zeros(X_train.shape[1])  # for our training

    alphas = \
        minimize(likelihood_alphas, alphas_0,
                 (X_train, y_train[measured_val + "_x"], y_train[measured_val + "_y"], True),
                 method='Nelder-Mead')['x']

    bias = np.array([0])
    bias = minimize(water_model_loss, bias, (X_train, train_villages, af_obs_train, is_obs_train, alphas, True))['x']

    train_losses = water_model_loss(bias, X_train, train_villages, af_obs_train, is_obs_train, alphas)
    test_losses = water_model_loss(bias, X_test, test_villages, af_obs_test, is_obs_test, alphas)

    return {"train_losses": train_losses, "test_losses": test_losses, 'bias': bias, 'fi': alphas}


def simulation_loss_per_agg_level(data_df, agg_size, train_size=0.7):
    '''

    :param data_df: should have columns for X, true_y, local_y, variance
    :param obs_df:
    :param agg_size:
    :param train_size:
    :return:
    '''
    n_samples = len(data_df)
    X = data_df.drop(columns=['true_y', 'local_y', 'variance'])
    n_features = X.shape[1]
    n_villages = n_samples // agg_size
    village_ids = list(range(n_villages))
    villages = agg_size * village_ids + [n_villages + 1] * (n_samples % agg_size)
    random.shuffle(villages)
    data_df['village_id'] = villages

    train_village_ids = set(random.sample(village_ids, math.floor(train_size * len(village_ids))))
    train_indices = data_df['village_id'].isin(train_village_ids)
    test_indices = ~train_indices

    X_train = X[train_indices]
    true_y_train = data_df['true_y'][train_indices]
    local_y_train = data_df['local_y'][train_indices]
    variances_train = data_df['variance'][train_indices]
    train_villages = data_df['village_id'][train_indices]

    X_test = X[test_indices]
    true_y_test = data_df['true_y'][test_indices]
    local_y_test = data_df['local_y'][test_indices]
    variances_test = data_df['variance'][test_indices]
    test_villages = data_df['village_id'][test_indices]

    # optimize alphas, optimize bias, get loss whole model and get loss sigmas
    alphas_0 = np.zeros(n_features)

    alphas = minimize(likelihood_alphas, alphas_0,
                      (X_train, local_y_train, true_y_train, True),
                      method='Nelder-Mead')['x']
    beta_0 = np.array([0])
    beta = minimize(water_model_loss, beta_0, (X_train, train_villages, local_y_train, true_y_train, alphas, True))['x']

    v_pred_train = generate_variances(X_train, alphas, beta)
    v_loss_train = mean_squared_error(variances_train, v_pred_train)

    v_pred_test = generate_variances(X_test, alphas, beta)
    v_loss_test = mean_squared_error(variances_test, v_pred_test)

    train_losses = \
        water_model_loss(beta, X_train, train_villages, local_y_train, true_y_train, alphas)['reg']

    test_losses = \
        water_model_loss(beta, X_test, test_villages, local_y_test, true_y_test, alphas)['reg']

    # correlation variances and diffrences
    # y_diff_test = np.square(local_y_test - true_y_test) #TODO add correlation variances and differences

    return {"train_losses": train_losses, "test_losses": test_losses, 'beta': beta, 'fi': alphas,
            "variance_loss_train": v_loss_train, "variance_loss_test": v_loss_test}


def gen_samples(X, feature_weights, beta, mu=None, obs_lower_bound=0, obs_upper_bound=100):
    n_samples = X.shape[0]
    variances = generate_variances(X, feature_weights, beta=beta)
    if mu is None:
        mu = np.random.randint(obs_lower_bound, obs_upper_bound, n_samples)
    assert (len(mu) == n_samples)
    obs_no_clip = np.random.normal(mu, np.sqrt(variances))
    obs = np.clip(obs_no_clip, obs_lower_bound, obs_upper_bound)
    return obs, mu, variances


def calc_model_predictions(df, true_y_col="true_y", reported_y_col='local_y',
                           estimated_variance_col="estimated_variance",
                           prediction_unit_col='village'):
    '''
    Calculates 0-1 loss on the village division means.
    :param df:
    :return:
    '''

    df = df.assign(var_inverse=1 / df[estimated_variance_col])

    total_weights_per_pu = df[[prediction_unit_col, 'var_inverse']].groupby(prediction_unit_col).sum()
    total_weights_per_pu = total_weights_per_pu.rename(columns={'var_inverse': 'weights_sum'})
    df = df.merge(total_weights_per_pu, on=prediction_unit_col)
    df['sample_contribution'] = df['var_inverse'] * df['local_y'] / df['weights_sum']

    means_df = df[[prediction_unit_col, true_y_col, reported_y_col]].groupby(prediction_unit_col).mean()
    model_pred = df[[prediction_unit_col, 'sample_contribution']].groupby(prediction_unit_col).sum()
    results_df = pd.DataFrame({"true_mean": means_df[true_y_col], "naive_mean": means_df[reported_y_col],
                               'model_mean': model_pred['sample_contribution']})
    return results_df


def count_accuracy(results_df):
    '''Results df is the output of calc_model_predictions'''
    diff_model = np.abs(results_df['true_mean'] - results_df['model_mean'])
    diff_naive = np.abs(results_df['true_mean'] - results_df['naive_mean'])
    return sum(diff_model < diff_naive) / len(results_df)


def results_mse(results_df):
    '''Results df is the output of calc_model_predictions'''
    diff_model = np.square(results_df['true_mean'] - results_df['model_mean'])/len(results_df)
    diff_naive = np.square(results_df['true_mean'] - results_df['naive_mean'])/len(results_df)
    return (diff_model,diff_naive)
