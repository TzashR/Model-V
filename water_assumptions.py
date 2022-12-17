'''
This is a notebook that analyses the water datasets
'''
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest as ztest
from matplotlib import pyplot as plt

# %% recreate lishtot df
with open("Water/swahili.xlsx", 'rb') as f:
    local = pd.read_excel(f)

with open("Water/english.xlsx", 'rb') as f:
    israeli = pd.read_excel(f)

lishtot_cols = ['Lishtot_score1','Lishtot_score2','Lishtot_score3','Lishtot_score4','Lishtot_score5']
israeli['lishtot'] = np.mean(israeli[lishtot_cols], axis = 1)
local['lishtot'] = np.mean(local[lishtot_cols], axis = 1)

israeli['pu'] = israeli['village_name_standard']

#%% Variance of "true" average lishtot score in prediction units
merged = local.merge(israeli[['KEY','Lishtot Average','pu']],on = 'KEY')
grouped_var = merged.groupby('pu').var()
grouped_count = merged.groupby('pu').count()


#%% compare variance in taking repeated measures
ax = israeli[lishtot_cols].var(axis = 1).hist(bins=15)
ax.set_ylabel("Household count")
ax.set_xlabel("Variance")
ax.set_title("Distribution of Lishtot Variance in Households - University Team")
plt.show()

ax = local[lishtot_cols].var(axis = 1).hist(bins=15)
ax.set_ylabel("Household count")
ax.set_xlabel("Variance")
ax.set_title("Distribution of Lishtot Variance in Households - Water Ambassadors")

plt.show()
