# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:12:41 2021

@author: Joe.WozniczkaWells
"""

import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Define competition. Choose one of the following:
#   - IPL: ipl
#   - T20 Blast: ntb
comp = "ipl"

# Original format csvs - need these to get information on match including the
# winner and outcome of the toss, the player of the match, the winning team
# and the winning margin (in runs or wickets)
csv_original = zipfile.ZipFile("Data/{0}_csv.zip".format(comp))

# "Ashwin format" csvs - easier to get ball-by-ball information from these
# than from original format ones
csv_ashwin = zipfile.ZipFile("Data/{0}_csv2.zip".format(comp))

files_original = csv_original.namelist()
#print(files_original)

files_ashwin = csv_ashwin.namelist()
#print(files_ashwin)

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

#%% Non-IPL-specific

# Create boolean variable based on whether a wicket was taken or not
df_all_matches['wicket'] = np.where(df_all_matches.wicket_type.notna(),1,0)

# Try creating rolling total runs scored for a batsman
df_all_matches['bat_career_runs'] = (df_all_matches
              .groupby('striker').runs_off_bat.cumsum())

# Check
df_all_matches[df_all_matches['striker']=='SC Ganguly'][['striker','runs_off_bat','bat_career_runs']]
# Tick

# Create rolling balls faced for a batsman
df_all_matches['bat_career_balls_faced'] = (df_all_matches[
        df_all_matches.wides.isna()].groupby('striker').cumcount()+1)
df_all_matches['bat_career_balls_faced'] = (df_all_matches
              .groupby('striker')['bat_career_balls_faced']
              .fillna(method='bfill')
              .fillna(method='ffill'))

# Check
df_all_matches[df_all_matches['striker']=='SC Ganguly'][['striker','runs_off_bat','bat_career_balls_faced']]

# Create rolling count of dots faced and 1s, 2s, 3s, 4s, 5s and 6s scored.

for i in list(range(0,7)):
    df_all_matches['bat_career_{0}s'.format(i)] = (df_all_matches
                   [(df_all_matches['wides'].isna()) &
                    (df_all_matches['runs_off_bat'] == i)]
                   .groupby('striker').cumcount()+1)
    df_all_matches['bat_career_{0}s'.format(i)] = (df_all_matches
                   .groupby('striker')['bat_career_{0}s'.format(i)]
                   .fillna(method='ffill').fillna(0))



# Check
#test = df_all_matches[df_all_matches['striker']=='SC Ganguly'][['striker','runs_off_bat','bat_career_balls_faced','bat_career_0s',
#              'bat_career_1s','bat_career_2s','bat_career_3s','bat_career_4s','bat_career_6s']]
#test.to_csv('test.csv')
# Tick

# Use similar code to create runs scored, balls faced and 0s, 1s, 2s, 3s, 4s, 5s and 6s scored in each innings
# Try creating rolling total runs scored for a batsman
df_all_matches['bat_innings_runs'] = (df_all_matches
              .groupby(['match_id','striker'])
              .runs_off_bat.cumsum())

# Check
#test = df_all_matches[df_all_matches['striker']=='SC Ganguly'][['match_id','striker','runs_off_bat','bat_innings_runs']]
#test.to_csv('test.csv')
# Tick

# Create rolling balls faced for a batsman
df_all_matches['bat_innings_balls_faced'] = (df_all_matches[
        df_all_matches.wides.isna()].groupby(['match_id','striker']).cumcount()+1)
df_all_matches['bat_innings_balls_faced'] = (df_all_matches
              .groupby(['match_id','striker'])['bat_innings_balls_faced']
              .fillna(method='bfill')
              .fillna(method='ffill')
              )

# Check
#test = df_all_matches[df_all_matches['striker']=='SC Ganguly'][['match_id','striker','wides','bat_innings_balls_faced']]
#test.to_csv('test.csv')
# Tick

# Create rolling count of dots faced and 1s, 2s, 3s, 4s and 6s scored. 
# 5s could be assumed to be 1s with 4 overthrows because of the scarcity of 3s 
# scored in IPL history (629 from 200664 balls).

for i in list(range(0,7)):
    # number of runs scored off the bat
    df_all_matches['bat_innings_{0}s'.format(i)] = (df_all_matches
                   [(df_all_matches['wides'].isna()) &
                    (df_all_matches['runs_off_bat'] == i)]
                   .groupby(['match_id','striker']).cumcount()+1)
    df_all_matches['bat_innings_{0}s'.format(i)] = (df_all_matches
                   .groupby(['match_id','striker'])['bat_innings_{0}s'.format(i)]
                   .fillna(method='ffill').fillna(0))
    # as a proportion of balls faced
    df_all_matches['bat_innings_{0}s_prop'.format(i)] = (
            df_all_matches['bat_innings_{0}s'.format(i)]/
            df_all_matches['bat_innings_balls_faced'])

# Check
#test = df_all_matches[df_all_matches['striker']=='SC Ganguly'][['striker','runs_off_bat','bat_innings_balls_faced','bat_innings_0s','bat_innings_1s','bat_innings_2s','bat_innings_3s','bat_innings_4s','bat_innings_6s']]
#test.to_csv('test.csv')
# Tick

## Try creating number of runs scored in last 5 innings
#df_all_matches['bat_runs_last5inns'] = df_all_matches.groupby(['match_id','striker']).tail(1)['bat_innings_runs'].transform(lambda x: x.rolling(5).sum())
#df_all_matches['bat_runs_last5inns'] = df_all_matches.groupby(['match_id','striker']).tail(1)['bat_innings_runs'].rolling(5, min_periods=1).sum()

## Check
#test = df_all_matches[df_all_matches['striker']=='SC Ganguly'][['match_id','striker','bat_innings_runs','bat_innings_balls_faced','bat_runs_last5inns']]
#test.to_csv('test.csv')
## No - not sure what this is doing now

# Create runs scored off the bat, extras (inc the different types of extras) and wickets by innings
inns_sum_list = ['runs_off_bat','extras','wides','noballs','byes','legbyes','penalty','wicket']

for i in inns_sum_list:
    df_all_matches['inns_{0}'.format(i)] = (df_all_matches
                   .groupby(['match_id','innings'])['{0}'.format(i)]
                   .cumsum())
    df_all_matches['inns_{0}'.format(i)] = (df_all_matches
                   .groupby(['match_id','innings'])['inns_{0}'.format(i)]
                   .fillna(method='ffill').fillna(0))

#test = df_all_matches[df_all_matches['match_id']==1254086]
#test.to_csv('test.csv')

# Create batting order number
## Melt to get striker and non-striker in the same column then sort and drop
## duplicates.
df_bat_order = (pd.melt(df_all_matches, 
                        id_vars = ['match_id','innings','ball'], 
                        value_vars = ['striker','non_striker'])
    .sort_values(by = ['match_id','innings','ball'])
    .drop_duplicates(subset = ['match_id','innings','value'])
    [['match_id','innings','value']])

## Group by match_id and innings and assign order    
df_bat_order['bat_order'] = (df_bat_order
            .groupby(['match_id','innings'])
            .value
            .transform(lambda x : pd.factorize(x)[0]+1))

## Merge batting order on for striker and non-striker
df_all_matches = (df_all_matches
                  .merge(df_bat_order,
                         left_on = ['match_id','innings','striker'],
                         right_on = ['match_id','innings','value'])
                  .drop(columns=['value'])
                  .rename(columns={'bat_order': 'bat_order_striker'})
                  .merge(df_bat_order,
                         left_on = ['match_id','innings','non_striker'],
                         right_on = ['match_id','innings','value'])
                  .drop(columns=['value'])
                  .rename(columns={'bat_order': 'bat_order_non_striker'}))
## There must be a better way to do this but I couldn't figure it out
                  
# Categorise bat_order_striker and bat_order_non_striker
df_all_matches['bat_order_striker_cat'] = np.where(
        df_all_matches['bat_order_striker']<=3,'TopOrder',
        np.where(df_all_matches['bat_order_striker']<=6,'MiddleOrder',
                 np.where(df_all_matches['bat_order_striker']<=8,
                          'LowerMiddleOrder','LowerOrder')))

df_all_matches['bat_order_non_striker_cat'] = np.where(
        df_all_matches['bat_order_non_striker']<=3,'TopOrder',
        np.where(df_all_matches['bat_order_non_striker']<=6,'MiddleOrder',
                 np.where(df_all_matches['bat_order_non_striker']<=8,
                          'LowerMiddleOrder','LowerOrder')))

# Group by match_id, innings and striker to get information on individual
# innings by batters.
df_all_matches_bat_inns = (df_all_matches
                           .sort_values(['match_id','innings','ball'])
                           .groupby(['match_id','innings','striker']).tail(1))

#df_all_matches_bat_inns.to_csv("df_all_matches_bat_inns.csv")

# Variables to use for clustering:
# Numeric (to be standardised before clustering): bat_innings_runs, bat_innings_balls_faced, bat_innings_0s, bat_innings_1s,
#   bat_innings_2s, bat_innings_3s, bat_innings_4s, bat_innings_5s, bat_innings_6s, bat_order_striker
# Categorical (to be one-hot encoded): bat_order_striker_cat

numeric_features = ['bat_innings_runs','bat_innings_balls_faced',
                    # Don't think I should include the bat_innings_... fields because they will be correlated with the ..._prop ones
                    #'bat_innings_0s','bat_innings_1s','bat_innings_2s',
                    #'bat_innings_3s','bat_innings_4s','bat_innings_5s',
                    #'bat_innings_6s',
                    'bat_innings_0s_prop','bat_innings_1s_prop',
                    'bat_innings_2s_prop','bat_innings_3s_prop',
                    'bat_innings_4s_prop','bat_innings_5s_prop',
                    'bat_innings_6s_prop',
                    'bat_order_striker']
categorical_features = ['bat_order_striker_cat']

# One-hot encode categorical features
df_kmeans_unstandardised = pd.get_dummies(df_all_matches_bat_inns,
                                          columns = categorical_features,
                                          drop_first = True)

# Standardise numeric features
data_to_standardise = df_all_matches_bat_inns[numeric_features]
scaler = StandardScaler().fit(data_to_standardise)

df_kmeans_standardised = df_kmeans_unstandardised.copy()
standardised_columns = scaler.transform(data_to_standardise)
df_kmeans_standardised[numeric_features] = standardised_columns

get_dummies_cols = []
for cat in categorical_features:
    get_dummies_cols = get_dummies_cols + [col for col in df_kmeans_standardised if cat in col]

df_kmeans_standardised = df_kmeans_standardised[numeric_features+get_dummies_cols]

# To do next:
# - get initial clustering solution
# - compare to input variable - Grand Index type thing
# - does this tell us anything?
# - play around with number of clusters

# Get k-means solutions for 1 cluster to 20 clusters, saving sse for each
max_nclus = 20

kmeans_kwargs = {}
kmeans_kwargs['init'] = 'random'
kmeans_kwargs['n_init'] = 20
kmeans_kwargs['max_iter'] = 500
kmeans_kwargs['random_state'] = 1

sse = []
for k in range(1,max_nclus+1):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df_kmeans_standardised)
    sse.append(kmeans.inertia_)

#Use elbow method to select optimal number of clusters
kl = KneeLocator(range(1,max_nclus+1),sse,curve='convex',direction='decreasing')
nclus = kl.elbow
print(str(nclus) + ' clusters')

# Get k-means solution for optimum number of clusters
kmeans = KMeans(n_clusters=nclus, **kmeans_kwargs)
kmeans.fit(df_kmeans_standardised)

# Add cluster numbers to df_all_matches_bat_inns
df_clustered = df_all_matches_bat_inns.copy()
df_clustered['inns_cluster'] = kmeans.predict(df_kmeans_standardised)
df_clustered['bat_innings_strike_rate'] = 100*df_clustered['bat_innings_runs']/df_clustered['bat_innings_balls_faced']

# Add one-hot encoded columns to df_all_matches_bat_inns
df_clustered = pd.get_dummies(df_clustered,
                              columns = categorical_features + ['batting_team','year'],
                              drop_first = False)

# Calculate means and indices
## Define list of features to be profiled
prof_features = ['bat_innings_runs', 'bat_innings_balls_faced','bat_innings_strike_rate',
                 'bat_innings_0s', 'bat_innings_0s_prop', 'bat_innings_1s',
                 'bat_innings_1s_prop', 'bat_innings_2s', 'bat_innings_2s_prop',
                 'bat_innings_3s', 'bat_innings_3s_prop', 'bat_innings_4s',
                 'bat_innings_4s_prop', 'bat_innings_5s', 'bat_innings_5s_prop',
                 'bat_innings_6s', 'bat_innings_6s_prop', 'bat_order_striker',
                 'bat_order_striker_cat_TopOrder', 
                 'bat_order_striker_cat_MiddleOrder',
                 'bat_order_striker_cat_LowerMiddleOrder',
                 'bat_order_striker_cat_LowerOrder',
                 'batting_team_Chennai Super Kings',
                 'batting_team_Deccan Chargers', 'batting_team_Delhi Capitals',
                 'batting_team_Delhi Daredevils', 'batting_team_Gujarat Lions',
                 'batting_team_Kings XI Punjab', 
                 'batting_team_Kochi Tuskers Kerala',
                 'batting_team_Kolkata Knight Riders', 
                 'batting_team_Mumbai Indians', 'batting_team_Pune Warriors', 
                 'batting_team_Punjab Kings', 'batting_team_Rajasthan Royals', 
                 'batting_team_Rising Pune Supergiant', 
                 'batting_team_Rising Pune Supergiants',
                 'batting_team_Royal Challengers Bangalore',
                 'batting_team_Sunrisers Hyderabad']

# Calculate cluster/variable and variable means
df_clustered_melt = pd.melt(df_clustered, id_vars = ['inns_cluster'],
                            value_vars = prof_features)

df_clustered_melt['cluster_mean'] = df_clustered_melt.groupby(['inns_cluster','variable'])['value'].transform('mean')
df_clustered_melt['variable_mean'] = df_clustered_melt.groupby(['variable'])['value'].transform('mean')
df_clustered_melt['index'] = round(100*df_clustered_melt['cluster_mean']/df_clustered_melt['variable_mean'])

# Drop duplicates and pivot to get Grand Index
GI = (df_clustered_melt[['inns_cluster','variable','variable_mean','cluster_mean','index']].
      drop_duplicates(subset = ['inns_cluster','variable']).
      pivot_table(index=['variable','variable_mean'],columns='inns_cluster'))






 


