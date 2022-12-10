# %%
import numpy as np
from scipy.optimize import minimize

from Generic_Calcs import likelihood_alphas, generate_none_spatial_data, water_model_loss,simulation_loss_per_agg_level



#%% experiment : same data set, different aggregations
repeats_per_agg = 3
n_samples = 5000
n_features = 1
ds = generate_none_spatial_data(n_features, n_samples, min_obs_village=10, max_obs_village=10, feature_weights=np.array([1]),
                               split_train_test=False,
                               train_test_ratio=0.3, max_sigma=100)['df']
betas_av = []
betas_std = []
variance_loss_av_train = []
variance_loss_av_test = []
variance_loss_std_train = []
variance_loss_std_test= []
model_loss_av_train = []
model_loss_av_test = []
model_loss_std_train = []
model_loss_std_test = []
naiv_loss_av = []
max_agg = 10
step = 2
agg_lvls = list(range(1,max_agg,step))
for agg in agg_lvls:
    print(f"agg {agg} out of {max_agg}")
    agg_betas = []
    agg_variance_loss_test = []
    agg_variance_loss_train = []
    agg_model_loss_test = []
    agg_model_loss_train = []
    agg_naive_loss_train = []
    agg_naive_loss_test = []
    for i in range(repeats_per_agg):
        r = simulation_loss_per_agg_level(ds,agg)
        agg_betas.append(r['beta'])
        agg_variance_loss_test.append(r['variance_loss_test'])
        agg_variance_loss_train.append(r['variance_loss_train'])
        agg_model_loss_train.append(r['train_losses'][1])
        agg_naive_loss_train.append(r['train_losses'][0])
        agg_naive_loss_test.append(r['test_losses'][0])
        agg_model_loss_test.append(r['test_losses'][1])

    betas_av.append(np.mean(agg_betas))
    betas_std.append(np.std(agg_betas))
    variance_loss_av_train.append(np.mean(agg_variance_loss_train))
    variance_loss_av_test.append(np.mean(agg_variance_loss_test))
    variance_loss_std_train.append(np.std(agg_variance_loss_train))
    variance_loss_std_test.append(np.std(agg_variance_loss_test))
    model_loss_av_train.append(np.mean(agg_model_loss_train))
    model_loss_av_test.append(np.mean(agg_model_loss_test))
    model_loss_std_train.append(np.std(agg_model_loss_train))
    model_loss_std_test.append(np.std(agg_model_loss_test))

# %%

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
