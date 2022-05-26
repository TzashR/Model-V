import math
import random

import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

from Generic_Calcs import likelihood_alphas, predict_variances


# %%
def generate_data(n_features, n_samples, weights=None):
    '''
    Generates data similar to the water reports. We can use this to test our model where the goal is to learn
    the weights from the features and obs returned
    :param n_features:
    :param n_samples:
    :param weights:
    :return:
    '''
    if weights is None:
        weights = np.random.uniform(0, 4.6 / n_features, n_features)
        # This is because I limit the feature size to 1 for all features, and I dont want to get sigma^2 > 100
    assert n_features == len(weights)
    features = np.random.rand(n_samples, n_features)  # array n_samples X n_features of ~U(0,1)
    sigmas = np.exp(features @ weights)
    israeli_obs = np.random.randint(0, 95, n_samples)
    obs = np.random.normal(israeli_obs, sigmas)
    return {'features': features, 'israeli_obs': israeli_obs, 'af_obs': obs, 'sigmas': sigmas, 'weights': weights}


def accuracy_over_n_samples(samples_lower, samples_upper, step, n_features, T):
    ns = [n for n in range(samples_lower, samples_upper, step)]
    alphas_0 = np.array([0, 0, 0, 0])

    av_acc = []
    for n in ns:
        acc = 0
        for i in range(T):
            data = generate_data(n_features, n)
            X = data['features']
            is_obs = data['israeli_obs']
            af_obs = data['af_obs']
            weights = data['weights']
            res = minimize(likelihood_alphas, alphas_0, (X, af_obs, is_obs, True), method='Nelder-Mead')
            dist = np.linalg.norm(res['x'] - weights * 2)  # *2 because it works for some reason. should debug this
            acc += dist / T
        av_acc.append(acc)
    plt.plot(ns, av_acc)
    plt.title("Accuracy vs n samples")
    plt.xlabel("n samples")
    plt.ylabel("Accuracy")
    plt.show()


def accuracy_over_n_features(f_lower, f_upper, step, n_samples, T):
    ns = [n for n in range(f_lower, f_upper, step)]
    av_acc = []
    for n in ns:
        print(n)
        acc = 0
        alphas_0 = np.zeros(n)
        for i in range(T):
            data = generate_data(n, n_samples)
            X = data['features']
            is_obs = data['israeli_obs']
            af_obs = data['af_obs']
            weights = data['weights']
            res = minimize(likelihood_alphas, alphas_0, (X, af_obs, is_obs, True), method='Nelder-Mead')
            dist = np.linalg.norm(res['x'] - weights * 2)  # *2 because it works for some reason. should debug this
            acc += dist / T
        av_acc.append(acc)
    plt.plot(ns, av_acc)
    plt.title("Accuracy vs n features")
    plt.xlabel("n features")
    plt.ylabel("Accuracy")
    plt.show()


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
    bias = minimize(loss_model, bias, (X_train, train_villages, af_obs_train, is_obs_train, alphas, True))['x']

    train_losses = loss_model(bias, X_train, train_villages, af_obs_train, is_obs_train, alphas)
    test_losses = loss_model(bias, X_test, test_villages, af_obs_test, is_obs_test, alphas)

    return {"train_losses": train_losses, "test_losses": test_losses, 'bias': bias, 'fi': alphas}


def loss_model(bias, X, villages_division, obs, is_obs, alphas, for_optim=False):
    model_loss_reg = 0
    naive_loss_reg = 0
    model_loss_root = 0
    naive_loss_root = 0

    villages_unique = np.unique(villages_division)
    variances = predict_variances(X, alphas, bias)

    res_dic = {}

    for v in villages_unique:
        indices = (villages_division == v)
        variances_v = variances[indices]

        # print(f" indices = {indices},v = {v}")
        obs_v = obs[indices]
        is_obs_v = is_obs[indices]
        true_av = is_obs_v.mean()

        # model_weights = np.reciprocal(variances_v) + bias
        model_weights = np.reciprocal(variances_v)
        model_av = np.dot(model_weights, obs_v) / sum(model_weights)
        model_loss_reg += (model_av - true_av) ** 2
        model_loss_root += model_loss_reg * np.sqrt(sum(indices))

        naive_av = obs_v.mean()
        naive_loss_reg += (naive_av - true_av) ** 2
        naive_loss_root += naive_loss_reg * np.sqrt(sum(indices))

    naive_loss_reg /= len(villages_unique)
    model_loss_reg /= len(villages_unique)
    model_loss_root /= sum(np.sqrt(villages_division.value_counts()))
    naive_loss_root /= sum(np.sqrt(villages_division.value_counts()))

    res_dic['reg'] = (naive_loss_reg, model_loss_reg)
    res_dic['root'] = (naive_loss_root, model_loss_root)

    if for_optim:
        return model_loss_reg
    return res_dic


# %% recreate lishtot df
with open("Water/swahili.xlsx", 'rb') as f:
    local = pd.read_excel(f)

with open("Water/english.xlsx", 'rb') as f:
    israeli = pd.read_excel(f)

local['Af STD'] = local[['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].std(
    axis=1)
local['Af mean'] = local[
    ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].mean(axis=1)
israeli['Is STD'] = israeli[
    ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].std(axis=1)
israeli['Is mean'] = israeli[
    ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].mean(axis=1)

local = local.rename(
    columns={'Lishtot_score1': 'Af 1', 'Lishtot_score2': 'Af 2', 'Lishtot_score3': 'Af 3', 'Lishtot_score4': 'Af 4',
             'Lishtot_score5': 'Af 5'})
israeli = israeli.rename(
    columns={'Lishtot_score1': 'Is 1', 'Lishtot_score2': 'Is 2', 'Lishtot_score3': 'Is 3', 'Lishtot_score4': 'Is 4',
             'Lishtot_score5': 'Is 5'})
lishtot = pd.merge(local[["KEY", "Af 1", "Af 2", "Af 3", "Af 4", "Af 5", "Af mean", "Af STD"]],
                   israeli[["KEY", "Is 1", "Is 2", "Is 3", "Is 4", "Is 5", "Is mean", "Is STD"]], on='KEY')
# lishtot['diff'] = ((lishtot['Is mean'] - lishtot['Af mean'])**2)/2
local['measurement_time'] = round((local['endtime'] - local['starttime']).dt.total_seconds() / 60, 2)  # in minutes

# TODO keep doing this. Remove features with very low variance
scaler = MinMaxScaler()
numericals_df = df[list(set(df.columns).difference(set(cat_features)))]
numericals_df.drop(columns=['KEY'], inplace=True)
numericals_df = numericals_df.fillna(0)
scaled_nums = scaler.fit_transform(numericals_df)
vars = scaled_nums.var(axis=0)
vars_df = pd.DataFrame({"feature": numericals_df.columns, "variance": vars})

over_fitting_features = ["Total_chlorine_values", "free_chlorine_values", "water_quantities"]

# %% create data for model

measured_val = 'Lishtot Average'  # total_alkalinity' total_hardness_values,'Lishtot Average
df = local[['KEY', 'geo-Altitude', 'sex', 'village_id', 'age', 'household_residents', 'number_children_household',
            'water_container', 'container_material', 'liters', 'payment', 'room', 'water_taken_options',
            'source_question', 'water_treatment', 'treatment_time_of_day', 'water_safety', 'water_smell',
            'taste_details', 'water_quantities', 'Total_chlorine_values', 'free_chlorine_values',
            'total_hardness_values', 'total_alkalinity', 'pH_values', 'Lishtot Average']]

df['lishtot_std'] = local[['Af 1', 'Af 2', 'Af 3', 'Af 4', 'Af 5']].std(axis=1)
cat_features = ['sex', 'water_container', 'container_material', 'payment', 'room',
                'water_taken_options', 'source_question', 'water_treatment', 'treatment_time_of_day', 'water_safety',
                'water_smell', 'taste_details', 'water_quantities']

cat_features = list(set(cat_features).difference(set(over_fitting_features)))
df.drop(columns=over_fitting_features, inplace=True)

df = df.merge(israeli[['KEY', measured_val]], on="KEY")

df = df.drop(columns=["KEY"])

df = df.fillna('0')
obs = df[[measured_val + "_x", measured_val + "_y"]]
df = df.drop([measured_val + "_y"], axis=1)

df = pd.get_dummies(df, columns=cat_features)
df = df.astype(float)
obs = obs.astype(float)

##### normal model application
# %%
villages = df["village_id"]
train_village_ids = {1, 2, 5}
train_indices = villages.isin(train_village_ids)
train_villages = df[train_indices]['village_id']

test_indices = ~train_indices
test_villages = df[test_indices]['village_id']

df.drop(columns=['village_id'], inplace=True)

X_train = df[train_indices]
X_test = df[test_indices]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = obs[train_indices]
is_obs_train = y_train[measured_val + "_y"]
af_obs_train = y_train[measured_val + "_x"]
y_test = obs[test_indices]

is_obs_test = y_test[measured_val + "_y"]
af_obs_test = y_test[measured_val + "_x"]

alphas_0 = np.zeros(X_train.shape[1])  # for our training

alphas = \
    minimize(likelihood_alphas, alphas_0, (X_train, y_train[measured_val + "_x"], y_train[measured_val + "_y"], True),
             method='Nelder-Mead')['x']

bias = np.array([0])
bias = minimize(loss_model, bias, (X_train, train_villages, af_obs_train, is_obs_train, alphas, True))['x']

train_losses = loss_model(bias, X_train, train_villages, af_obs_train, is_obs_train, alphas)
test_losses = loss_model(bias, X_test, test_villages, af_obs_test, is_obs_test, alphas)

print(
    f"Train: naive loss = {train_losses['reg'][0]}, model loss = {train_losses['reg'][1]} \n root naive =  {train_losses['root'][0]}, root model =  {train_losses['root'][1]} ")
print(
    f"Test: naive loss = {test_losses['reg'][0]}, model loss = {test_losses['reg'][1]} \n root naive =  {test_losses['root'][0]}, root model =  {test_losses['root'][1]} ")

#
# bias = \
# minimize(loss_model, bias_0, (data['f_train'], data['train_villages'], data['obs_train'], data['is_train'], alphas),
#          method='Nelder-Mead')['x']


#######


# %%
# accuracy_over_n_samples(100, 3000, 200, 4)
# accuracy_over_n_features(3, 12, 1, 1500)
# %%
# data = generate_data(4, 1000000)
# X = data['features']
# is_obs = data['israeli_obs']
# af_obs = data['af_obs']
# weights = data['weights']
# alphas_0 = np.array([0, 0, 0, 0])
# %%
# res1 = minimize(likelihood_alphas, alphas_0, (X, af_obs, is_obs, True), method='Nelder-Mead')
#
# # %% for likelihood
# data2 = data[['Is mean', 'Af mean', 'Af STD', 'measurement_time', 'age', 'liters']]
# data2 = data2.dropna(subset=data2.columns)
# #%%
# is_samples = data2['Is mean']
# af_samples = data2['Af mean']
# distances = np.square(is_samples - af_samples)
# features = data2[['Af STD', 'measurement_time', 'age', 'liters']]
#
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)
# X_train, X_test, y_train, y_test = train_test_split(features, distances, test_size=0.25, random_state=42)
#
# # %% costume function
# alphas_0 = np.array([0.25, 0.25, 0.25, 0.25])
#
# res = minimize(likelihood_alphas, alphas_0, (X_train, y_train, True))
#
# %% grid search for sigma
# grid = np.linspace(0.1,100,400)
# best_sigma = None
# best_l = -float('inf')
#
# for sigma in grid:
#     l = likelihood_fixed_sigma(sigma,y_train,True)
#     if l > best_l:
#         best_l = l
#         best_sigma = sigma

# %% train test comparison
train_loss_naive, train_loss_model = train_losses['reg']
test_loss_naive, test_loss_model = test_losses['reg']
losses_df = pd.DataFrame(
    {"Naive Loss": [train_loss_naive, test_loss_naive], "Model Loss": [train_loss_model, test_loss_model]},
    index=["Train", "Test"])

losses_df.plot(kind="bar")
plt.title("Model Vs Naive loss")
plt.show()

# %% Distribution of veracities
variances = predict_variances(df.fillna(0), alphas)
# all_samples['vars']  = variances
variances.plot.density()
plt.show()

plt.title("Distribution of observation variances")
cuts = pd.cut(variances, bins=15)  ##Maybe use qcut?
ax = cuts.value_counts().plot(kind='barh')
ax.bar_label(ax.containers[0])
plt.xlabel("Number of observations")
plt.ylabel("Variance bin range")
plt.show()

# %% feature importance
fi_df = pd.DataFrame({'col': df.columns, "importance": alphas})
top_features_reliable = fi_df.sort_values('importance')
top_features_unreliable = fi_df.sort_values('importance', ascending=False)

top_features_reliable.iloc[:20].plot.barh(x='col', y='importance', title="Feature importance - reliable")
plt.show()
top_features_unreliable.iloc[:20].plot.barh(x='col', y='importance', title="Feature importance - unreliable")
plt.show()
# %%agg levels
repeats = 10
aggs = list(range(1, 100,2))
biases_av = []
biases_std = []
model_losses_av = []
model_losses_std = []
naive_losses = []
for agg in aggs:
    print(agg)
    agg_biases = []
    agg_model_losses = []
    agg_naive_losses = []

    for i in range(repeats):
        res = loss_per_agg_level(df, obs, agg, measured_val, train_size=0.7)
        agg_biases.append(res['bias'])
        agg_model_losses.append(res['test_losses']['reg'][1])
        agg_naive_losses.append(res['test_losses']['reg'][0])

    biases_av.append(np.mean(agg_biases))
    biases_std.append(np.std(agg_biases))
    model_losses_av.append(np.mean(agg_model_losses))
    model_losses_std.append(np.std(agg_model_losses))
    naive_losses.append(np.mean(agg_naive_losses))

# %%
# plot losses over aggs
green_patch = mpatches.Patch(color='green', label='Naive Loss')
red_patch = mpatches.Patch(color='red', label='Model loss')
plt.legend(handles=[green_patch, red_patch])

plt.plot(aggs, naive_losses, color='green')
plt.plot(aggs, model_losses_av, color='red')
plt.title(f"Loss vs Aggregation Level")
plt.xlabel("Number of households in village")
plt.show()

# %%plot std
plt.title("Loss STD vs Aggregation Level")
plt.plot(aggs, model_losses_std, color='blue')
plt.show()

# %%Plot bias
green_patch = mpatches.Patch(color='green', label='Bias Average')
red_patch = mpatches.Patch(color='red', label='Bias STD')
plt.legend(handles=[green_patch, red_patch])

plt.title("Bias vs Aggregation Level")
plt.plot(aggs, biases_av, color='green')
plt.plot(aggs, biases_std, color='red')
plt.show()
