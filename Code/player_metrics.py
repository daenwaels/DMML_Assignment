# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:38:36 2021

@author: Dean.Wales
"""

# %% import modules and custom functions

import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns
from Code.functions import ballbyballvars


# %% read file
# Define competition. Choose one of the following:
#   - IPL: ipl
#   - T20 Blast: ntb
comp = "ipl"

# Original format csvs - need these to get information on match including the
# winner and outcome of the toss, the player of the match, the winning team
# and the winning margin (in runs or wickets)
csv_original = zipfile.ZipFile("./Data/{0}_csv.zip".format(comp))

# "Ashwin format" csvs - easier to get ball-by-ball information from these
# than from original format ones
csv_ashwin = zipfile.ZipFile("./Data/{0}_csv2.zip".format(comp))

files_original = csv_original.namelist()

files_ashwin = csv_ashwin.namelist()

# README.txt is the first file, the rest are csvs. Ignore README.txt and loop
# through csvs.

# When trying to read in the original format csvs I run into a problem because
# the first few ('info') rows have fewer columns than the 'ball' rows. This
# can be solved by defining the column names for the columns we want from these
# files and giving the rest numeric names that will later be removed.
spare_cols = list(map(str, list(range(4,16))))
cols = ['Type','Var','Value'] + spare_cols

# Create empty dataframe
df_info_long = pd.DataFrame()

for filename in files_original[1:872]:
    # Read in info from csv_original
    df_info = pd.read_csv(csv_original.open(filename), names = cols, 
                          skiprows = 1)
    df_info['match_id'] = filename.replace('.csv','')
    df_info = df_info[df_info['Type'] == 'info'][['Var','Value']]
    df_info.loc[0,'Var'] = 'home_team'
    df_info.loc[1,'Var'] = 'away_team'
    df_info_long = df_info_long.append(df_info)
    
# Read in ball-by-ball from csv_ashwin
df_all_matches = pd.read_csv(csv_ashwin.open("all_matches.csv"))

#%% IPL-specific data cleaning

# DtypeWarning: Columns (1) have mixed types.Specify dtype option on import or set low_memory=False.
print(df_all_matches.dtypes)
# I think the warning came about because read_csv processes files in batches
# and in some batches season will have been seen as an object (e.g. 2007/08)
# and in some it will have been seen as an integer (e.g. 2015). Convert 
# start_date to date and check whether any seasons have spanned years.

df_all_matches['start_date'] = pd.to_datetime(df_all_matches['start_date'])
df_all_matches['year'] = pd.DatetimeIndex(df_all_matches['start_date']).year

print(df_all_matches.groupby(['season','year']).size())

# All seasons contain matches from a single year and there has never been more
# than one season in a year, therefore year can be used  and season can be 
# discarded.

# Some teams' names have changed throughout the existence of the IPL. Standardise these.
print(pd.value_counts(df_all_matches.batting_team))

# Kings XI Punjab/Punjab Kings: Kings XI Punjab renamed to Punjab Kings - use current name
# Delhi Daredevils/Delhi Capitals: Delhi Daredevils renamed to Delhi Capitals - use current name
# Rising Pune Supergiants/Rising Pune Supergiant: 's' removed after first season, disbanded after second - use most recent

df_all_matches = df_all_matches.replace("Kings XI Punjab","Punjab Kings")
df_all_matches = df_all_matches.replace("Delhi Daredevils","Delhi Capitals")
df_all_matches = df_all_matches.replace("Rising Pune Supergiants",
                                        "Rising Pune Supergiant")

# After more thought, I've decided to keep batting innings played by only the current 8 teams, since Deccan Chargers last 
# played in 2012 and KTK, PW, GL and RPS played no more than 3 seasons each.

teams = ['Chennai Super Kings','Delhi Capitals','Kolkata Knight Riders','Mumbai Indians','Punjab Kings','Rajasthan Royals',
         'Royal Challengers Bangalore','Sunrisers Hyderabad']

df_all_matches = (df_all_matches[df_all_matches['batting_team'].isin(teams)].reset_index())

#%% Non-IPL-specific
    
# Remove records with innings > 2 from the data. These will have come from super overs, which are not counted towards
# players' career stats. proc stands for processed
df_all_matches_proc = df_all_matches[df_all_matches['innings']<=2].reset_index()


def ballbyballvars(df):
    
    # Create boolean variable based on whether a wicket was taken or not
    df['wicket'] = np.where(df.wicket_type.notna(),1,0)
    
    # Create rolling total runs scored for a batsman
    df['bat_career_runs'] = df.groupby('striker').runs_off_bat.cumsum()
    
    # Create rolling balls faced for a batsman
    df['bat_career_balls_faced'] = (df[df.wides.isna()].groupby('striker').
      cumcount()+1)
    df['bat_career_balls_faced'] = (df
      .groupby('striker')['bat_career_balls_faced'].fillna(method='bfill')
      .fillna(method='ffill'))
    
    # Create rolling count of dots faced and 1s, 2s, 3s, 4s, 5s and 6s scored
    # in  career.
    for i in list(range(0,7)):
        df['bat_career_{0}s'.format(i)] = (df[(df['wides'].isna()) & 
          (df['runs_off_bat'] == i)].groupby('striker').cumcount()+1)
        df['bat_career_{0}s'.format(i)] = (df
           .groupby('striker')['bat_career_{0}s'.format(i)]
           .fillna(method='ffill').fillna(0))
        
    # Create rolling total runs scored for a batsman
    df['bat_innings_runs'] = (df.groupby(['match_id','striker']).runs_off_bat
      .cumsum())
    
    # Create rolling balls faced for a batsman
    df['bat_innings_balls_faced'] = (df[df.wides.isna()].groupby(
            ['match_id','striker']).cumcount()+1)
    df['bat_innings_balls_faced'] = (df.groupby(['match_id','striker'])
        ['bat_innings_balls_faced'].fillna(method='ffill')
        .fillna(0))
    
    # Create rolling count of dots faced and 1s, 2s, 3s, 4s, 5s and 6s scored
    # in innings.
    for i in list(range(0,7)):
        # number of runs scored off the bat
        df['bat_innings_{0}s'.format(i)] = (df[(df['wides'].isna()) & 
          (df['runs_off_bat'] == i)].groupby(['match_id','striker'])
            .cumcount()+1)
        df['bat_innings_{0}s'.format(i)] = (df.groupby(['match_id','striker'])
            ['bat_innings_{0}s'.format(i)].fillna(method='ffill').fillna(0))
        # as a proportion of balls faced
        df['bat_innings_{0}s_prop'.format(i)] = (np.where(df['bat_innings_balls_faced']>0,
                df['bat_innings_{0}s'.format(i)]/
                df['bat_innings_balls_faced'],0))

    # Create runs scored off the bat, extras (inc the different types of extras) and wickets by innings
    inns_sum_list = ['runs_off_bat','extras','wides','noballs','byes',
                     'legbyes','penalty','wicket']
    
    for i in inns_sum_list:
        df['inns_{0}'.format(i)] = (df.groupby(['match_id','innings'])
            ['{0}'.format(i)].cumsum())
        df['inns_{0}'.format(i)] = (df.groupby(['match_id','innings'])
            ['inns_{0}'.format(i)].fillna(method='ffill').fillna(0))
        
    # Create batting order number
    ## Melt to get striker and non-striker in the same column then sort and drop
    ## duplicates.
    df_bat_order = (pd.melt(df, id_vars = ['match_id','innings','ball'], 
                            value_vars = ['striker','non_striker'])
        .sort_values(by = ['match_id','innings','ball'])
        .drop_duplicates(subset = ['match_id','innings','value'])
        [['match_id','innings','value']])
    
    ## Group by match_id and innings and assign order    
    df_bat_order['bat_order'] = (df_bat_order.groupby(['match_id','innings'])
        .value.transform(lambda x : pd.factorize(x)[0]+1))
    
    ## Merge batting order on for striker and non-striker
    df = (df.merge(df_bat_order, left_on = ['match_id','innings','striker'],
                   right_on = ['match_id','innings','value'])
        .drop(columns=['value'])
        .rename(columns={'bat_order': 'bat_order_striker'})
        .merge(df_bat_order, left_on = ['match_id','innings','non_striker'],
               right_on = ['match_id','innings','value'])
        .drop(columns=['value'])
        .rename(columns={'bat_order': 'bat_order_non_striker'}))
    ## There must be a better way to do this but I couldn't figure it out
                      
    # Categorise bat_order_striker and bat_order_non_striker
    df['bat_order_striker_cat'] = pd.Categorical(np.where(df['bat_order_striker']<=3,
      'Top',np.where(df['bat_order_striker']<=6,'Middle',
                          np.where(df['bat_order_striker']<=8,
                                   'Lower','Tail'))),['Top','Middle','Lower','Tail'])
    
    df['bat_order_non_striker_cat'] = pd.Categorical(np.where(df['bat_order_non_striker']<=3,
      'Top',np.where(df['bat_order_non_striker']<=6,'Middle',
                          np.where(df['bat_order_non_striker']<=8,
                                   'Lower','Tail'))),['Top','Middle','Lower','Tail'])
    
    df['bat_innings_strike_rate'] = 100*df['bat_innings_runs']/df['bat_innings_balls_faced']
    
    return df


# Use ballbyballvars function to create extra variables
df_all_matches_proc = ballbyballvars(df_all_matches_proc)

# Group by match_id, innings and striker to get information on individual
# innings by batters.
batting_metrics = (df_all_matches_proc
                           .sort_values(['match_id','innings','ball'])
                           .groupby(['match_id','innings','striker']).tail(1)
                           [['striker', 'batting_team', 'innings', 'wicket', 
                             'bat_innings_runs','bat_innings_balls_faced', 'runs_off_bat',
                             'bat_innings_0s',
                             'bat_innings_1s',
                             'bat_innings_2s',
                             'bat_innings_3s',
                             'bat_innings_4s',
                             'bat_innings_5s',
                             'bat_innings_6s',
                             'inns_extras','inns_wides',
                             'inns_noballs','inns_byes','inns_legbyes',
                             'inns_penalty','inns_wicket','bat_order_striker',
                             'bat_order_striker_cat']])


# %% nearest neighbours classification

# summarise to player, batting_team and innings level

batting_metrics = batting_metrics.groupby(['striker', 'batting_team']
                                          ).agg(
                                              {
                                                  'bat_innings_runs' : sum,
                                                  'bat_innings_balls_faced' : sum,
                                                  'wicket' : sum,
                                                  'bat_innings_0s' : sum,
                                                  'bat_innings_1s' : sum,
                                                  'bat_innings_2s' : sum,
                                                  'bat_innings_3s' : sum,
                                                  'bat_innings_4s' : sum,
                                                  'bat_innings_5s' : sum,
                                                  'bat_innings_6s' : sum,
                                                  'innings' : 'count',
                                                  'bat_order_striker' : lambda x: x.value_counts().index[0],
                                                  'bat_order_striker_cat' : lambda x: x.value_counts().index[0]
                                                  }
                                              ).reset_index()

batting_metrics = batting_metrics[batting_metrics['bat_innings_balls_faced'] != 0]   


# calculate batting metrics


batting_metrics['strike_rate'] = (batting_metrics['bat_innings_runs'] / batting_metrics['bat_innings_balls_faced']) * 100
     
batting_metrics['ran_runs'] = batting_metrics['bat_innings_runs'] - ((batting_metrics['bat_innings_6s'] * 6) + (batting_metrics['bat_innings_4s'] * 4))
                                         
batting_metrics['activity_rate'] = batting_metrics['ran_runs'] / (batting_metrics['bat_innings_balls_faced'] - ( batting_metrics['bat_innings_6s'] + batting_metrics['bat_innings_4s']))

batting_metrics['boundary_rate'] = (batting_metrics['bat_innings_6s'] + batting_metrics['bat_innings_4s']) / batting_metrics['bat_innings_balls_faced']

batting_metrics['average'] = batting_metrics['bat_innings_runs'] / batting_metrics['innings']

batting_metrics['consistency'] = batting_metrics['bat_innings_runs'] / batting_metrics['wicket']

batting_metrics['consistency'] = batting_metrics['consistency'].replace([np.nan], 0)

batting_metrics['consistency'] = np.where(batting_metrics['consistency'] == np.inf, batting_metrics['bat_innings_runs'], batting_metrics['consistency'])

for i in list(range(0,7)):
    batting_metrics['bat_innings_{0}s_prop'.format(i)] = (np.where(batting_metrics['bat_innings_balls_faced'] > 0,
                                                      batting_metrics['bat_innings_{0}s'.format(i)]/
                                                      batting_metrics['bat_innings_balls_faced'],0))


# select for features to be used

batting_metrics = batting_metrics[['striker', 'batting_team',
                                   'bat_innings_runs', 'bat_innings_balls_faced',                                   
                                   'ran_runs', 
                                   'wicket',
                                   'bat_innings_0s', 'bat_innings_0s_prop',
                                   'bat_innings_1s', 'bat_innings_1s_prop',
                                   'bat_innings_2s', 'bat_innings_2s_prop',
                                   'bat_innings_3s', 'bat_innings_3s_prop',
                                   'bat_innings_4s', 'bat_innings_4s_prop',
                                   'bat_innings_5s', 'bat_innings_5s_prop',
                                   'bat_innings_6s', 'bat_innings_6s_prop',
                                   'strike_rate', 'activity_rate', 'boundary_rate', 
                                   'average', 'consistency',
                                   'bat_order_striker_cat']]

batting_metrics = batting_metrics[['striker', 'batting_team',
                                   'bat_innings_runs', 'bat_innings_balls_faced',                                   
                                   'ran_runs', 
                                   'wicket',
                                   'strike_rate', 'activity_rate', 'boundary_rate', 
                                   'average', 'consistency',
                                   'bat_order_striker_cat']]

batting_metrics = batting_metrics.set_index(['striker', 'batting_team'])


# drop NaNs

batting_metrics = batting_metrics.dropna()

batting_metrics['bat_order_striker_cat'] = batting_metrics['bat_order_striker_cat'].map({'Top': 0, 'Middle': 1, 'Lower': 2, 'Tail': 3})

# split dataset

X = batting_metrics.drop(['bat_order_striker_cat'], axis = 1)

y = batting_metrics['bat_order_striker_cat']


# encode prediction variable

#lab_enc = LabelEncoder()

#y = lab_enc.fit_transform(y)


# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train_raw = X_train
X_test_raw = X_test
y_train_raw = y_train
y_test_raw = y_test


# scale features

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# train knn model

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

knn.predict(X_test)
knn.score(X_test, y_test)


knn.predict(X_train)
knn.score(X_train, y_train)


# train knn cv model

knn_cv = KNeighborsClassifier(n_neighbors = 3)

cv_scores = cross_val_score(knn_cv, X, y, cv=5)

print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


# tuning knn model

knn2 = KNeighborsClassifier()

param_grid = {'n_neighbors' : np.arange(1, 25)}

knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

knn_gscv.fit(X, y)

knn_gscv.best_params_
knn_gscv.best_score_




# model matrix

models = []

models.append(('KNN', 'KNN.pk1', KNeighborsClassifier()))
models.append(('RC', 'RC.pk1', RidgeClassifier(probability=True)))
models.append(('SGD', 'SGD.pk1', SGDClassifier()))
models.append(('DT', 'DT.pk1', DecisionTreeClassifier()))
models.append(('RF', 'RF.pk1', RandomForestClassifier()))

names = []
train_accuracy = []
train_error = []
test_accuracy = []
test_error = []


# set kfold params

kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)


# initiate loop for scores

for name, save, model in models:
    model_cv_accuracy = cross_val_score(model, X_train, y_train,  cv = kfold, scoring = 'accuracy')
    train_accuracy.append(model_cv_accuracy.mean())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy.append(accuracy_score(y_test, y_pred))
    names.append(name)
    
    
    
scores = pd.DataFrame({'Name': names, 'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy})

print(scores)




# train rf model

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf.predict(X_test)
rf.score(X_test, y_test)


rf.predict(X_train)
rf.score(X_train, y_train)


# tune knn model

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)

random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)

print(rf_random.best_params_)


rf = RandomForestClassifier(n_estimators = 400, max_features = 'sqrt', max_depth = 300)


# fit tuned model

rf.fit(X_train, y_train)

X_test_rf_pred = rf.predict(X_test)
print(X_test_rf_pred)
rf.score(X_test, y_test)


rf.predict(X_train)
rf.score(X_train, y_train)


# cross validation

kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)

model_cv_accuracy = cross_val_score(rf, X_train, y_train, cv = kfold, scoring = 'accuracy')

print(model_cv_accuracy.mean())



# amend dataframe 

X_test_raw['actual'] = y_test
X_test_raw['predicted'] = X_test_rf_pred

X_test_raw['distance'] = abs(X_test_raw['actual'] - X_test_raw['predicted'])

conditions = [
    X_test_raw['distance'] == 0,
    X_test_raw['distance'] == 1,
    X_test_raw['distance'] >= 2
    ]

values = ['exact', 'close', 'incorrect']

X_test_raw['result'] = np.select(conditions, values)



X_test_raw['actual'] = X_test_raw['actual'].map({0: 'Top', 1: 'Middle', 2: 'Lower', 3: 'Tail'})

X_test_raw['predicted'] = X_test_raw['predicted'].map({0: 'Top', 1: 'Middle', 2: 'Lower', 3: 'Tail'})



# calculate results

X_test_raw_percent = round(len(X_test_raw[(X_test_raw['result'] == 'exact') | (X_test_raw['result'] == 'close')]) / len(X_test_raw) * 100, 1)

print(X_test_raw_percent)


X_test_raw_filt = X_test_raw[(X_test_raw['bat_innings_balls_faced'] >= 50)]

X_test_raw_filt_percent = round(len(X_test_raw_filt[(X_test_raw_filt['result'] == 'exact') | (X_test_raw_filt['result'] == 'close')]) / len(X_test_raw_filt) * 100, 1)

print(X_test_raw_filt_percent)


# set categorical data

X_test_raw['actual'] = pd.Categorical(X_test_raw['actual'], ['Top', 'Middle', 'Lower', 'Tail'])

X_test_raw['result'] = pd.Categorical(X_test_raw['result'], ['exact', 'close', 'incorrect'])


X_test_raw = X_test_raw.sort_values(['actual', 'result'])


X_test_raw_filt['actual'] = pd.Categorical(X_test_raw_filt['actual'], ['Top', 'Middle', 'Lower', 'Tail'])

X_test_raw_filt['result'] = pd.Categorical(X_test_raw_filt['result'], ['exact', 'close', 'incorrect'])


X_test_raw_filt = X_test_raw_filt.sort_values(['actual', 'result'])



# set params for mosaic plot

props = {}

props[('Top', 'exact')]={'facecolor':'green', 'edgecolor':'white'}
props[('Middle', 'exact')]={'facecolor':'green', 'edgecolor':'white'}
props[('Lower', 'exact')]={'facecolor':'green', 'edgecolor':'white'}
props[('Tail', 'exact')]={'facecolor':'green', 'edgecolor':'white'}

props[('Top', 'close')]={'facecolor':'yellow', 'edgecolor':'white'}
props[('Middle', 'close')]={'facecolor':'yellow', 'edgecolor':'white'}
props[('Lower', 'close')]={'facecolor':'yellow', 'edgecolor':'white'}
props[('Tail', 'close')]={'facecolor':'yellow', 'edgecolor':'white'}

props[('Top', 'incorrect')]={'facecolor':'red', 'edgecolor':'white'}
props[('Middle', 'incorrect')]={'facecolor':'red', 'edgecolor':'white'}
props[('Lower', 'incorrect')]={'facecolor':'red', 'edgecolor':'white'}
props[('Tail', 'incorrect')]={'facecolor':'red', 'edgecolor':'white'}


# unfiltered dataframe

pd.crosstab(X_test_raw['actual'], X_test_raw['result'])

mosaic(X_test_raw, ['actual', 'result'], gap = 0.01, properties = props)



# filtered dataframe

pd.crosstab(X_test_raw_filt['actual'], X_test_raw_filt['result'])

mosaic(X_test_raw_filt, ['actual', 'result'], gap = 0.01, properties = props)
