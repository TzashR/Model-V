# %%
import numpy as np
from scipy.optimize import minimize

from Generic_Calcs import likelihood_alphas, generate_none_spatial_data, water_model_loss


# %%
def experiment(n_features,n_samples,max_obs_village,min_obs_village,repeats_per_lvl, agg_lvls):
    biases_av = []
    biases_std = []
    model_losses_av = []
    model_losses_std = []
    naive_losses = []

    data = generate_none_spatial_data(n_features, n_samples, min_obs_village=min_obs_village,
                                      max_obs_village=max_obs_village,
                                      split_train_test=True, max_sigma=100, train_test_ratio=0.3)

    for agg in agg_lvls:
        for i in range(repeats_per_lvl):
        # train_distances = np.square(data['is_train'] - data['obs_train'])


        alphas_0 = np.zeros(n_features)  # for our training
        # train by our model (max likelihood)
        alphas = minimize(likelihood_alphas, alphas_0, (data['f_train'], data['obs_train'], data['is_train'], True),
                          method='Nelder-Mead')['x']

        bias_0 = np.array([0])
        bias = minimize(loss_model, bias_0, (data['f_train'], data['train_villages'], data['obs_train'], data['is_train'], alphas),
                 method='Nelder-Mead')['x']
        biases.append(bias[0])

        loss = get_ds_loss(data['f_test'], data['test_villages'], alphas, data['obs_test'], data['is_test'])
        alt_loss =  get_ds_loss(data['f_test'], data['test_villages'], alphas, data['obs_test'], data['is_test'], bias = bias)

        # predicted_variances = np.exp(data['f_train'] @ alphas)
        # true_variances = data['variances_train']

        # print(f"loss bias = 0 : {loss}")
        # print(f"loss bias = {bias} : {alt_loss}")

        av_no_b_loss += loss[0]/n_tests
        av_model_loss += alt_loss[0]/n_tests
        av_naive_loss +=loss[1]/n_tests

    biases = np.array(biases)
    b_mean = biases.mean()
    b_std = biases.std()
    return av_naive_loss, av_no_b_loss,av_model_loss,b_mean,b_std

def plot_expriments_results(results,x,param:str):
    from matplotlib import patches as mpatches
    from matplotlib import pyplot as plt
    green_patch = mpatches.Patch(color='green', label='Naive Loss')
    red_patch = mpatches.Patch(color='red', label='Model no bias loss')
    blue_patch = mpatches.Patch(color='blue', label='Model with bias loss')
    plt.legend(handles=[green_patch, blue_patch,red_patch])

    naive_losses = [r[0] for r in results]
    no_bias_losses = [r[1] for r in results]
    model_losses = [r[2] for r in results]
    av_bs = [r[3] for r in results]
    b_stds = [r[4] for r in results]

    plt.plot(x,naive_losses, color = 'green')
    plt.plot(x,no_bias_losses, color = 'red')
    plt.plot(x,model_losses, color = 'blue')
    plt.title(f"Losses vs {param}")
    plt.show()


    green_patch = mpatches.Patch(color='green', label='Bias mean')
    red_patch = mpatches.Patch(color='red', label='Bias SD')
    plt.legend(handles=[green_patch,red_patch])
    plt.title(f"bias mean and bias STD vs {param}")

    plt.plot(x,av_bs, color = 'green')
    plt.plot(x,b_stds, color = 'red')
    plt.show()


min_obs = 10
max_obs = 30
n_samples = 3000
n_features = 5
n_tests = 5

#%% plot n_samples
try_samples = [500 *i for i in range(1,31)]
results = []
for n in try_samples:
    print(n)
    results.append(experiment(n_tests = n_tests,n_features = n_features,n_samples=n,max_obs_village=max_obs,min_obs_village=min_obs))
plot_expriments_results(results,[i for i in range(1,31)],"n_samples / 500")

#%%
fresults = []
try_features = [i for i in range(1,17)]
for n in try_features:
    print(n)
    fresults.append(experiment(n_tests = n_tests,n_features = n,n_samples=n_samples,max_obs_village=max_obs,min_obs_village=min_obs))
plot_expriments_results(fresults,try_features,"n_features")

#%%
max_obs = [3*i for i in range(1,12)]
obs_results = []
for n in max_obs:
    print(n)
    obs_results.append(experiment(n_tests = n_tests,n_features = n_features,n_samples=n_samples,max_obs_village=n,min_obs_village=max(n-5,2)))
plot_expriments_results(obs_results,max_obs,"max obs per village")


# %%
# n_samples = 10000
# israeli_obs = np.ones(n_samples)*50
# features = np.ones(n_samples)
# af_obs = np.random.normal(50, np.log(100),n_samples)
# alphas_0 = np.array([[0]])
# alphas = minimize(likelihood_alphas, alphas_0, (features, af_obs, israeli_obs, True),
#                              method='Nelder-Mead')['x']
