import numpy as np
from scipy.optimize import minimize

from Generic_Calcs import generate_variances, likelihood_alphas, generate_none_spatial_data


def test_gen_variances():
    # generate n_samples with n_features
    X = np.zeros((3, 1))
    X[0:3, 0] = [1, 2, 3]
    alphas = np.array([1])
    vars = generate_variances(X, alphas)
    assert (np.log(vars).flatten() == X.flatten()).all()


def test_learn_alphas():
    epsilon = 0.1
    n_samples = 1000

    # 1 discrete feature alpha = 1
    X = np.random.choice([0, 1], (n_samples, 1))
    true_alphas = np.ones(1)
    alphas_0 = np.zeros(1)
    vars = generate_variances(X, true_alphas)
    true_y = np.ones(n_samples) * 0.5
    reported_y = np.random.normal(true_y, np.sqrt(vars))

    alphas = minimize(likelihood_alphas, alphas_0,
                      (X, reported_y, true_y, True),
                      method='Nelder-Mead')['x']
    assert (np.absolute(alphas - true_alphas) < epsilon).all()
    learned_likelihood = likelihood_alphas(alphas, X, reported_y, true_y, return_minus=True)
    optimal_likelihood = likelihood_alphas(alphas_0, X, reported_y, true_y, return_minus=True)
    control_likelihood = likelihood_alphas(np.ones_like(alphas_0) * 0.5, X, reported_y, true_y, return_minus=True)

    # 1 continuous features alpha = 1
    X = np.random.rand(n_samples, 1)
    true_alphas = np.ones(1)
    alphas_0 = np.zeros(1)
    vars = generate_variances(X, true_alphas)
    true_y = np.ones(n_samples) * 0.5
    reported_y = np.random.normal(true_y, np.sqrt(vars))

    alphas = minimize(likelihood_alphas, alphas_0,
                      (X, reported_y, true_y, True),
                      method='Nelder-Mead')['x']
    assert (np.absolute(alphas - true_alphas) < epsilon).all()
    learned_likelihood = likelihood_alphas(alphas, X, reported_y, true_y, return_minus=True)
    optimal_likelihood = likelihood_alphas(alphas_0, X, reported_y, true_y, return_minus=True)
    control_likelihood = likelihood_alphas(np.ones_like(alphas_0) * 0.5, X, reported_y, true_y, return_minus=True)

    # 4  discrete features alpha = 1 (above 6 we start losing)
    n_features = 10
    n_samples = 10000
    X = np.random.choice([0, 1], (n_samples, n_features))
    true_alphas = np.ones(n_features)
    alphas_0 = np.zeros(n_features)
    alphas_0 = true_alphas
    vars = generate_variances(X, true_alphas)
    true_y = np.ones(n_samples) * 0.5
    reported_y = np.random.normal(true_y, np.sqrt(vars))

    alphas = minimize(likelihood_alphas, alphas_0,
                      (X, reported_y, true_y, True),
                      method='Nelder-Mead')['x']
    assert (np.absolute(alphas - true_alphas) < epsilon).all()
    learned_likelihood = likelihood_alphas(alphas, X, reported_y, true_y, return_minus=False)
    optimal_likelihood = likelihood_alphas(true_alphas, X, reported_y, true_y, return_minus=False)
    control_likelihood = likelihood_alphas(alphas_0, X, reported_y, true_y, return_minus=False)
    print(f"learned l = {learned_likelihood}, optimal_l = {optimal_likelihood}, control_l = {control_likelihood}")

    # random number of continuous features alpha = 1
    n_samples = 10 * 1000
    X = np.random.rand(n_samples, n_features)
    n_features = 4
    true_alphas = np.ones(n_features)
    alphas_0 = np.zeros(n_features)
    vars = generate_variances(X, true_alphas)
    true_y = np.ones(n_samples) * 0.5
    reported_y = np.random.normal(true_y, np.sqrt(vars))

    alphas = minimize(likelihood_alphas, alphas_0,
                      (X, reported_y, true_y, True),
                      method='Nelder-Mead')['x']
    assert (np.absolute(alphas - true_alphas) < epsilon).all()
    n_samples = 10 * 1000
    X = np.random.rand(n_samples, n_features)
    n_features = 4
    true_alphas = np.random.rand(n_features)
    alphas_0 = np.zeros(n_features)
    vars = generate_variances(X, true_alphas)
    true_y = np.ones(n_samples) * 0.5
    reported_y = np.random.normal(true_y, np.sqrt(vars))

    alphas = minimize(likelihood_alphas, alphas_0,
                      (X, reported_y, true_y, True),
                      method='Nelder-Mead')['x']
    assert (np.absolute(alphas - true_alphas) < epsilon).all()
    learned_likelihood = likelihood_alphas(alphas, X, reported_y, true_y, return_minus=True)
    optimal_likelihood = likelihood_alphas(alphas_0, X, reported_y, true_y, return_minus=True)
    control_likelihood = likelihood_alphas(np.ones_like(alphas_0) * 0.5, X, reported_y, true_y, return_minus=True)

    # random number of features random alphas


def n_features_sensitivity():
    ''' Testing this with no train test split'''
    n_samples = 10**3
    feature_ns = list(range(1, 8))
    obs_per_village = 4

    for n in feature_ns:
        weights = np.ones(n)*8

        data = generate_none_spatial_data(n_features=n, n_samples=n_samples, min_obs_village=obs_per_village,
                                          max_obs_village=obs_per_village, n_cat_features=0,feature_weights=weights)
        df = data['df']
        df = df.assign(village = data['villages_division'])
        X = df.drop(columns=['true_y','local_y','variance','village'])

        alphas_0 = np.ones(n)

        alphas= minimize(likelihood_alphas, alphas_0,
                          (X, df['local_y'], df['true_y'], True),
                          method='Nelder-Mead')['x']

        df = df.assign(estimated_variance = generate_variances(X,alphas))
         likelihood_alphas(alphas,X,df['local_y'],df['true_y'], False)








