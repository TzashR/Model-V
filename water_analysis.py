'''
This is a notebook that analyses the water datasets
'''
import pandas as pd
from statsmodels.stats.weightstats import ztest as ztest

# %% recreate lishtot df
with open("Water/swahili.xlsx", 'rb') as f:
    local = pd.read_excel(f)

with open("Water/english.xlsx", 'rb') as f:
    israeli = pd.read_excel(f)

# local['Af STD'] = local[['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].std(
#     axis=1)
# local['Af mean'] = local[
#     ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].mean(axis=1)
# israeli['Is STD'] = israeli[
#     ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].std(axis=1)
# israeli['Is mean'] = israeli[
#     ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].mean(axis=1)

# local = local.rename(
#     columns={'Lishtot_score1': 'Af 1', 'Lishtot_score2': 'Af 2', 'Lishtot_score3': 'Af 3', 'Lishtot_score4': 'Af 4',
#              'Lishtot_score5': 'Af 5'})
# israeli = israeli.rename(
#     columns={'Lishtot_score1': 'Is 1', 'Lishtot_score2': 'Is 2', 'Lishtot_score3': 'Is 3', 'Lishtot_score4': 'Is 4',
#              'Lishtot_score5': 'Is 5'})
# lishtot = pd.merge(local[["KEY", "Af 1", "Af 2", "Af 3", "Af 4", "Af 5", "Af mean", "Af STD"]],
#                    israeli[["KEY", "Is 1", "Is 2", "Is 3", "Is 4", "Is 5", "Is mean", "Is STD"]], on='KEY')
# lishtot['diff'] = ((lishtot['Is mean'] - lishtot['Af mean'])**2)/2

# remove unpaired key:
keys = local['KEY'].unique()
israeli = israeli[israeli['KEY'].isin(keys)]
israeli = israeli.sort_values(["KEY"])
local = local.sort_values(["KEY"])
# sort so keys would match
# %% Basic comparison
'''
1. 
    a. For Alkalinity, total_hardness, compare the mean and SD between the two sets.
    The output of this stage should be a table comparing the two values.  Do also a hypothesis test
    for the following h0: the average absolute difference between israeli measures and local measures is bigger than 0
    b. Specifically for Lishtot sensor, show that the sd of the 5 samples is similar between the two teams,
     do a paired t test
2. Show that the values of other measures (that aren't the measured values) are similar.
 Namely, For each measure do a coupled t test. The result for this stage should be a table shwowing for each measure it's
 p-value for the effect of difference between the teams.
'''

# q1a - Alkalinity
alks = israeli[["KEY", "total_alkalinity"]].merge(local[["KEY", "total_alkalinity"]], on=["KEY"])
alks.dropna(inplace=True)


# Because this is essentially order scale, I will transform the values to 0,1,2,3,4,5
values_dic = {0: 0, 40: 1, 80: 2, 120: 3, 180: 4, 240: 5}
alks_order = alks.replace(values_dic)

alks_order['diff'] = abs(alks_order["total_alkalinity_x"] - alks_order['total_alkalinity_y'])
#need to check if h0 holds for this 'diff' column