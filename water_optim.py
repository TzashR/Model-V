import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# %% recreate lishtot df
with open("Water/swahili.xlsx", 'rb') as f:
    local = pd.read_excel(f)

with open("Water/english.xlsx", 'rb') as f:
    israeli = pd.read_excel(f)

# %% create the training data frame
df = pd.DataFrame()

df = local[['geo-Altitude', 'sex', 'village_name', 'age', 'household_residents', 'number_children_household',
            'water_container', 'container_material', 'liters', 'payment', 'room', 'water_taken_options',
            'source_question', 'water_treatment', 'treatment_time_of_day', 'water_safety', 'water_smell',
            'taste_details', 'water_quantities', 'Total_chlorine_values', 'free_chlorine_values',
            'total_hardness_values', 'total_alkalinity', 'pH_values', 'Lishtot Average', 'Lishtot Average English']]

cat_features = ['sex', 'village_name', 'water_container', 'container_material', 'payment', 'room',
                'water_taken_options', 'source_question', 'water_treatment', 'treatment_time_of_day', 'water_safety',
                'water_smell', 'taste_details', 'water_quantities']

df['time_diff_hours'] = local.apply(lambda row: (row['endtime'] - row['starttime']).total_seconds() / 3600, axis=1)
df['lishtot_variance'] = local[
    ['Lishtot_score1', 'Lishtot_score2', 'Lishtot_score3', 'Lishtot_score4', 'Lishtot_score5']].var(axis=1)
df = df.fillna(0)
df = df.dropna()

y = ((df['Lishtot Average English'] - df['Lishtot Average']) ** 2) / 2

df = df.drop(['Lishtot Average English'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
# %% catboosting
model = CatBoostRegressor(n_estimators=3000)
train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(X_test, label=y_test, cat_features=cat_features)

model.fit(train_pool, verbose=100)
