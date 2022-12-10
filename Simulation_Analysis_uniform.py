import pandas as pd

df = pd.read_csv("full_optimal.csv")
from matplotlib import pyplot as plt

n_samples_bins = pd.IntervalIndex.from_tuples([(0, 100), (101, 500), (501, 2000), (2001, 10000), (10001, 50000)])
df = df.assign(n_samples_interval=pd.cut(df['n_samples'], n_samples_bins))

ind_cols = ['n_features', 'n_samples_interval', 'obs_per_village', 'feature_power']

# %% Figures with fixed params except number of features. Show number of features effects
grouping_cols = ['obs_per_village', 'n_samples_interval']
fixed_for_features = df[grouping_cols].drop_duplicates()


def row_spearman_features(row):
    row_df = df.loc[
        (df['obs_per_village'] == row['obs_per_village']) & (df['n_samples_interval'] == row['n_samples_interval'])]
    return row_df[['n_features', 'mse_model', 'count_accuracy']].corr(method='spearman')['n_features']['count_accuracy']


def row_pearson_features(row):
    row_df = df.loc[
        (df['obs_per_village'] == row['obs_per_village']) & (df['n_samples_interval'] == row['n_samples_interval'])]
    return row_df[['n_features', 'mse_model', 'count_accuracy']].corr(method='pearson')['n_features']['count_accuracy']


fixed_for_features = fixed_for_features.assign(
    features_spearman=fixed_for_features.apply(lambda row: row_spearman_features(row), axis=1))
fixed_for_features = fixed_for_features.assign(
    features_pearson=fixed_for_features.apply(lambda row: row_pearson_features(row), axis=1))

# %% same as above but with moving samples
grouping_cols = ['obs_per_village', 'n_features']
fixed_for_samples = df[grouping_cols].drop_duplicates()


def row_spearman_samples(row):
    row_df = df.loc[(df['obs_per_village'] == row['obs_per_village']) & (df['n_features'] == row['n_features'])]
    return row_df[['n_samples', 'mse_model', 'count_accuracy']].corr(method='spearman')['n_samples']['count_accuracy']


def row_pearson_samples(row):
    row_df = df.loc[(df['obs_per_village'] == row['obs_per_village']) & (df['n_features'] == row['n_features'])]
    return row_df[['n_samples', 'mse_model', 'count_accuracy']].corr(method='pearson')['n_samples']['count_accuracy']


fixed_for_samples = fixed_for_samples.assign(
    features_spearman=fixed_for_samples.apply(lambda row: row_spearman_samples(row), axis=1))
fixed_for_samples = fixed_for_samples.assign(
    features_pearson=fixed_for_samples.apply(lambda row: row_pearson_samples(row), axis=1))

# %% same as above moving obs
grouping_cols = ['n_samples_interval', 'n_features']
fixed_for_obs = df[grouping_cols].drop_duplicates()


def row_spearman_obs(row):
    row_df = df.loc[(df['n_samples_interval'] == row['n_samples_interval']) & (df['n_features'] == row['n_features'])]
    return row_df[['obs_per_village', 'mse_model', 'count_accuracy']].corr(method='spearman')['obs_per_village'][
        'count_accuracy']


def row_pearson_obs(row):
    row_df = df.loc[(df['n_samples_interval'] == row['n_samples_interval']) & (df['n_features'] == row['n_features'])]
    return row_df[['obs_per_village', 'mse_model', 'count_accuracy']].corr(method='pearson')['obs_per_village'][
        'count_accuracy']


fixed_for_obs = fixed_for_obs.assign(features_spearman=fixed_for_obs.apply(lambda row: row_spearman_obs(row), axis=1))
fixed_for_obs = fixed_for_obs.assign(features_pearson=fixed_for_obs.apply(lambda row: row_pearson_obs(row), axis=1))

# %% Figures for obs - Im not sure that pearson/spearman is good here because maybe the function is more like parabula


# %% stability analysis - STD of losses with same params


# %% Train a random forest predict count loss. Can we do that?
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

y = df['count_accuracy']
X = df[['n_samples', 'n_features', 'obs_per_village']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)

# %% Find sweet global sweet spots

useful_df = df.loc[df['count_accuracy_optimal'] > 0.5]
stressed_ss = pd.read_csv("sweet_spots_stress.csv")

# %% Run again 50 times for each sweet spot configuration to see what the odds for success in them is
useful_df_sss = stressed_ss.loc[stressed_ss['count_accuracy_optimal'] > 0.5]

g_useful_df_sss = useful_df_sss.groupby(['n_features', 'n_samples', 'obs_per_village']).mean()

# %% Analyze good configs
good_configs_df = g_useful_df_sss.query("count_accuracy > 0.5").reset_index()

# %%_features
ax = good_configs_df['n_features'].value_counts().sort_values().plot(kind="barh")
ax.set_ylabel("Number of Features")
ax.set_xlabel("Count")
plt.show()


# %%  bins of sample size
def get_n_samples(n):
    if 0 < n <= 100:
        return 100
    elif n <= 500:
        return 500
    elif n <= 2000:
        return 2000
    elif n <= 10000:
        return 10000
    else:
        return 50000


good_configs_df = good_configs_df.assign(
    rounded_n_samples=good_configs_df.apply(lambda row: get_n_samples(row['n_samples']), axis=1))

ax = good_configs_df['rounded_n_samples'].value_counts().sort_values().plot(kind="barh")
ax.set_ylabel("Number of Samples")
ax.set_xlabel("Count")
plt.show()

# %% obs per  village


# %% See if it's learnable using a forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

clf = RandomForestClassifier(max_depth=3, random_state=0)
from sklearn.metrics import confusion_matrix

X = df[['n_samples', 'obs_per_village', 'n_features']]
y = df['count_accuracy_optimal'] > 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()

y_pred_test = clf.predict(X_test)
