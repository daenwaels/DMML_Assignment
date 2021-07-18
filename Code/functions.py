# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 19:03:48 2021

@author: Joe.WozniczkaWells
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

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
    
    return df

def clusinns_kmeans(df,categorical_features,numeric_features,prof_features,init='k-means++',n_init=10,
                    max_iter=300,random_state=None,max_nclus=20):
    
    # One-hot encode categorical features
    df_kmeans_unstandardised = pd.get_dummies(df,columns = categorical_features,drop_first = True)
    
    # Standardise numeric features
    data_to_standardise = df[numeric_features]
    scaler = StandardScaler().fit(data_to_standardise)
    
    df_kmeans_standardised = df_kmeans_unstandardised.copy()
    standardised_columns = scaler.transform(data_to_standardise)
    df_kmeans_standardised[numeric_features] = standardised_columns
    
    get_dummies_cols = []
    for cat in categorical_features:
        get_dummies_cols = get_dummies_cols + [col for col in df_kmeans_standardised if cat in col]
    
    df_kmeans_standardised = df_kmeans_standardised[numeric_features+get_dummies_cols]
    ##
    kmeans_kwargs = {}
    kmeans_kwargs['init'] = init
    kmeans_kwargs['n_init'] = n_init
    kmeans_kwargs['max_iter'] = max_iter
    kmeans_kwargs['random_state'] = random_state
    ##
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
    df_clustered = df.copy()
    df_clustered['inns_cluster'] = kmeans.predict(df_kmeans_standardised)
    df_clustered['bat_innings_strike_rate'] = 100*df_clustered['bat_innings_runs']/df_clustered['bat_innings_balls_faced']
    
    # Add one-hot encoded columns to df_all_matches_bat_inns
    df_clustered = pd.get_dummies(df_clustered,
                                  columns = categorical_features + ['batting_team','year'],
                                  drop_first = False)
    
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
    
    return df_clustered, GI