import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from matplotlib import pyplot as plt
from Generic_Calcs import generate_variances, likelihood_alphas, generate_none_spatial_data, calc_model_predictions, \
    count_accuracy, mean_squared_error

from scipy.optimize import minimize

# %% recreate lishtot df
with open("Water/swahili.xlsx", 'rb') as f:
    local = pd.read_excel(f)

with open("Water/english.xlsx", 'rb') as f:
    israeli = pd.read_excel(f)


lishtot_cols = ['Lishtot_score1','Lishtot_score2','Lishtot_score3','Lishtot_score4','Lishtot_score5']
israeli['true_y'] = np.mean(israeli[lishtot_cols], axis = 1)
local['local_y'] = np.mean(local[lishtot_cols], axis = 1)

israeli['pu'] = israeli['village_name_standard']

#%% Variance of "true" average lishtot score in prediction units
merged = local.merge(israeli[['KEY','true_y','pu']],on = 'KEY')

# %% create the dataframe
df = merged[['geo-Altitude', 'sex', 'pu', 'age', 'household_residents', 'number_children_household',
            'water_container', 'container_material', 'liters', 'payment', 'room', 'water_taken_options',
            'source_question', 'water_treatment', 'treatment_time_of_day', 'water_safety', 'water_smell',
            'taste_details', 'Total_chlorine_values', 'free_chlorine_values',
            'total_hardness_values', 'total_alkalinity', 'pH_values', 'true_y','local_y']]

cat_features = ['sex', 'water_container', 'container_material', 'payment', 'room',
                'water_taken_options', 'source_question', 'water_treatment', 'treatment_time_of_day', 'water_safety', 'taste_details', 'water_quantities']

df['time_diff_hours'] = local.apply(lambda row: (row['endtime'] - row['starttime']).total_seconds() / 3600, axis=1)
df['lishtot_variance'] = local[lishtot_cols].var(axis=1)


#%%One hot encoding
cat_df = df[cat_features]

cat_df = cat_df.fillna('none')

num_df = df.drop(columns = cat_features)
num_df = num_df.fillna(0)

enc = OneHotEncoder(drop = 'first', sparse = False)
ohe = enc.fit_transform(cat_df)
ohe_df = pd.DataFrame(ohe)
full_df = pd.concat([ohe_df,num_df],axis = 1)

#%%Manually choose train prediction units
train_pus = ['shuleni','kyomu','mahuru','Kisangesangeni B','msufini','kalimani','puplic','lekure','majengo']
train_indices = df['pu'].isin(train_pus)
test_indices = ~train_indices
X = full_df.drop(columns=['true_y','pu'])
X['beta'] = 1

X_train = X[train_indices]
n_features = X.shape[1]

alphas_0 = np.zeros(n_features )*0.5

alphas = minimize(likelihood_alphas, alphas_0,
                  (X_train, full_df[train_indices]['local_y'], df[train_indices]['true_y'], True),
                  method='Nelder-Mead')['x']

df = full_df.assign(estimated_variance=generate_variances(X, alphas))

predictions = calc_model_predictions(df[test_indices], prediction_unit_col='pu')
accuracy = count_accuracy(predictions)
mse_naive = mean_squared_error(predictions['true_mean'], predictions['naive_mean'])
mse_model = mean_squared_error(predictions['true_mean'], predictions['model_mean'])


## Plot results


#%% Usefulness


diff_model = np.abs(predictions['true_mean'] - predictions['model_mean'])
diff_naive = np.abs(predictions['true_mean'] - predictions['naive_mean'])
useful_count = sum(diff_model<diff_naive)
not_useful_count = sum(diff_model>diff_naive)


overall_use_df = pd.DataFrame({'count':[useful_count,not_useful_count]}, index  = ['Model Useful','Model Not Useful'])
ax = overall_use_df.plot(kind="barh")
ax.set_title("Count of Prediction Units Where the Model Was Useful")
ax.set_xlabel("Count")
ax.get_legend().remove()
plt.bar_label(ax.containers[0])

plt.show()


#%% Predictions


predictions['True y'] = predictions['true_mean']
predictions['Naive Mean Estimation'] = predictions['naive_mean']
predictions['Model Estimation'] = predictions['model_mean']

ax = np.round(predictions,2).reset_index()[['True y','Naive Mean Estimation','Model Estimation']].plot(kind="barh")
ax.set_title("Estimation of y ")
ax.set_xlabel("y value")
ax.set_ylabel("Prediction Unit Serial Number")

plt.bar_label(ax.containers[0])
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[2])

plt.show()

#%% MSE
mse_df = pd.DataFrame({'MSE':[mse_model,mse_naive]}, index  = ['Model MSE','Naive Mean MSE'])
ax = mse_df.plot(kind="barh")
ax.set_title("Average MSE - Model vs Naive Mean")
ax.set_xlabel("Average MSE")
ax.get_legend().remove()


plt.bar_label(ax.containers[0])

plt.show()

