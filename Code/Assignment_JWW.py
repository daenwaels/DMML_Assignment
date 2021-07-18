# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 19:04:25 2021

@author: Joe.WozniczkaWells
"""

import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns
from functions import ballbyballvars, clusinns_kmeans
import matplotlib.patches as mpatches

# Define competition. Choose one of the following:
#   - IPL: ipl
#   - T20 Blast: ntb
comp = "ipl"

# Original format csvs - need these to get information on match including the
# winner and outcome of the toss, the player of the match, the winning team
# and the winning margin (in runs or wickets)
csv_original = zipfile.ZipFile("../Data/{0}_csv.zip".format(comp))

# "Ashwin format" csvs - easier to get ball-by-ball information from these
# than from original format ones
csv_ashwin = zipfile.ZipFile("../Data/{0}_csv2.zip".format(comp))

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

df_all_matches = (df_all_matches[df_all_matches['batting_team'].isin(['Chennai Super Kings','Delhi Capitals',
                                'Kolkata Knight Riders','Mumbai Indians','Punjab Kings','Rajasthan Royals',
                                'Royal Challengers Bangalore','Sunrisers Hyderabad'])].reset_index())

#%% Non-IPL-specific
    
# Remove records with innings > 2 from the data. These will have come from super overs, which are not counted towards
# players' career stats.
df_all_matches_proc = df_all_matches[df_all_matches['innings']<=2].reset_index()

# Use ballbyballvars function to create extra variables
df_all_matches_proc = ballbyballvars(df_all_matches_proc)

# Group by match_id, innings and striker to get information on individual
# innings by batters.
df_all_matches_bat_inns = (df_all_matches_proc
                           .sort_values(['match_id','innings','ball'])
                           .groupby(['match_id','innings','striker']).tail(1)
                           [['match_id','venue','innings','batting_team',
                             'bowling_team','striker','year',
                             'bat_innings_runs','bat_innings_balls_faced',
                             'bat_innings_0s','bat_innings_0s_prop',
                             'bat_innings_1s','bat_innings_1s_prop',
                             'bat_innings_2s','bat_innings_2s_prop',
                             'bat_innings_3s','bat_innings_3s_prop',
                             'bat_innings_4s','bat_innings_4s_prop',
                             'bat_innings_5s','bat_innings_5s_prop',
                             'bat_innings_6s','bat_innings_6s_prop',
                             'inns_runs_off_bat','inns_extras','inns_wides',
                             'inns_noballs','inns_byes','inns_legbyes',
                             'inns_penalty','inns_wicket','bat_order_striker',
                             'bat_order_striker_cat']])


#%% Have a go at visualising this data
# - Scatter plots:
#   - x-axis: balls faced, y-axis: runs scored, colour: team, shape: top/middle/lower-middle/lower order
#   - same but separate plots for each year, or a selection of years (e.g. 2008, 2014, 2020)
#   - x-axis: balls faced, y-axis: runs scored, colour: team, shape: 1st/2nd innings
# - Bar plots:
#   - x-axis: year, y-axis: proportion of 0s, 1s, ..., 6s
#   - same but split by team
# - Line plots:
#   - x-axis: year, y-axis: strike rate, colour: team
#   - x-axis: year, y-axis: strike rate, colour: team, line style: top/middle/lower-middle/lower order
#   - x-axis: year, y-axis: strike rate, colour: team, line style: 1st/2nd innings
#   - after I try k-means, plot innings as time series to show different 'shapes' of innings

order_order = ['Top','Middle','Lower','Tail']
    
# Scatter plot of balls v runs, split by team and order
grid_team_order_balls_runs = sns.FacetGrid(df_all_matches_bat_inns,col='batting_team',hue='bat_order_striker_cat',height=4,
                                           hue_order=order_order,col_wrap=4)
grid_team_order_balls_runs.map_dataframe(sns.scatterplot,x='bat_innings_balls_faced',y='bat_innings_runs')
grid_team_order_balls_runs.set_axis_labels("Balls faced","Runs scored")
grid_team_order_balls_runs.set_titles(col_template="{col_name}")
grid_team_order_balls_runs.add_legend()
grid_team_order_balls_runs._legend.set_title("Order")
grid_team_order_balls_runs.savefig("../Plots/grid_team_order_balls_runs.png")

# - Similar general pattern
# - Not much differentiation between top/middle/lower in terms of strike rates
# - Top order players play most of the longest innings for all teams
# - Some teams (RCB, CSK) have noticeable increase in gradient for the longest innings played by top order players
# - Some teams (DC, MI) have a less noticeable difference between the length of innings played by top and middle
#   order batters

# Line plot of strike rate over time, split by order
df_all_matches_bat_year_order = (df_all_matches_bat_inns.groupby(['year','bat_order_striker_cat'], as_index=False)
    ['bat_innings_balls_faced','bat_innings_runs'].sum())
df_all_matches_bat_year_order['strike_rate'] = (100*df_all_matches_bat_year_order['bat_innings_runs']/
                             df_all_matches_bat_year_order['bat_innings_balls_faced'])

fig,ax = plt.subplots()
sns.lineplot(data=df_all_matches_bat_year_order, x='year',y='strike_rate',hue='bat_order_striker_cat',
             hue_order=order_order)

df_all_matches_bat_year_team_order = (df_all_matches_bat_inns.groupby(['year','batting_team','bat_order_striker_cat'], 
                                                                      as_index=False)['bat_innings_balls_faced',
'bat_innings_runs'].sum())
df_all_matches_bat_year_team_order['strike_rate'] = (100*df_all_matches_bat_year_team_order['bat_innings_runs']/
                                  df_all_matches_bat_year_team_order['bat_innings_balls_faced'])

grid_team_order_strikerate = sns.FacetGrid(df_all_matches_bat_year_team_order,col='batting_team',hue='bat_order_striker_cat',
                                           hue_order=order_order,height=4,col_wrap=4)
grid_team_order_strikerate.map_dataframe(sns.lineplot,x='year',y='strike_rate')
grid_team_order_strikerate.set_axis_labels("Year","Strike rate")
grid_team_order_strikerate.set_titles(col_template="{col_name}")
grid_team_order_strikerate.add_legend()
grid_team_order_strikerate._legend.set_title("Order")
grid_team_order_strikerate.savefig("../Plots/grid_team_order_strikerate.png")

# - Patterns vary greatly, particularly strike rate for tailenders
# - Variable tail strike rate shows unpredictability of lower order hitting and license given to tailenders
# - Some teams (DC, KKR, MI) have similar strike rates for top and middle order
# - Some (CSK, RCB) have higher strike rate for middle order than top order
# - Some (PK, RR, SH) have higher strike rate for top order than middle

# Scatter plot of balls v runs, split by team, order and year
grid_team_order_balls_runs_year = sns.FacetGrid(df_all_matches_bat_inns,row='batting_team',hue='bat_order_striker_cat',
                                                col='year')
grid_team_order_balls_runs_year.map_dataframe(sns.scatterplot,x='bat_innings_balls_faced',y='bat_innings_runs')
grid_team_order_balls_runs_year.set_axis_labels("Balls faced","Runs scored")
grid_team_order_balls_runs_year.set_titles(col_template="{col_name}", row_template="{row_name}")
grid_team_order_balls_runs_year.add_legend()
grid_team_order_balls_runs_year._legend.set_title("Order")
grid_team_order_balls_runs_year.savefig("grid_team_order_balls_runs_year.png")

# Not sure this is any use - too much information

# Scatter plot of balls v runs, split by team, order and innings
grid_team_order_balls_runs_inns = sns.FacetGrid(df_all_matches_bat_inns,row='batting_team',hue='bat_order_striker_cat',
                                                col='innings')
grid_team_order_balls_runs_inns.map_dataframe(sns.scatterplot,x='bat_innings_balls_faced',y='bat_innings_runs')
grid_team_order_balls_runs_inns.set_axis_labels("Balls faced","Runs scored")
grid_team_order_balls_runs_inns.set_titles(col_template="Innings {col_name}", row_template="{row_name}")
grid_team_order_balls_runs_inns.add_legend()
grid_team_order_balls_runs_inns._legend.set_title("Order")
grid_team_order_balls_runs_inns.savefig("grid_team_order_balls_runs_inns.png")

# - Generally not a huge difference between 1st and 2nd innings
# - Some teams (RCB, CSK, SH) have more of an increase in scoring rate for top order batters when batting first
# - Across the board, maybe slightly fewer big innings by top order players when chasing

# Runs scored and balls faced can only tell us so much about tactics. For example, it's impossible to tell whether a batter
# dismissed for 20 off 15 started slowly and had just started to accelerate when they were dismissed or whether they had
# scored at a constant rate throughout. The way in which the innings was put together may help us to differentiate.

# Stacked bar plot of 0s, 1s, 2s, ..., 6s by team
df_all_matches_bat_team_balltypes = (df_all_matches_bat_inns.groupby(['batting_team'], as_index=False)['bat_innings_0s',
                           'bat_innings_1s','bat_innings_2s','bat_innings_3s','bat_innings_4s','bat_innings_5s',
                           'bat_innings_6s','bat_innings_balls_faced'].sum())

for i in list(range(0,7)):
        # as a proportion of balls faced
        df_all_matches_bat_team_balltypes['bat_innings_{0}s_prop'.format(i)] = (
                df_all_matches_bat_team_balltypes['bat_innings_{0}s'.format(i)]/
                df_all_matches_bat_team_balltypes['bat_innings_balls_faced'])

fig,ax = plt.subplots(figsize=(10,12))
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_0s_prop'],label=0,
       color='#C5C5F7')
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_1s_prop'],label=1,
       bottom=df_all_matches_bat_team_balltypes['bat_innings_0s_prop'],color='#8B8BEF')
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_2s_prop'],label=2,
       bottom=df_all_matches_bat_team_balltypes['bat_innings_0s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_1s_prop'],color='#5151E7')
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_3s_prop'],label=3,
       bottom=df_all_matches_bat_team_balltypes['bat_innings_0s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_1s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_2s_prop'],color='#EBBF12')
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_4s_prop'],label=4,
       bottom=df_all_matches_bat_team_balltypes['bat_innings_0s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_1s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_2s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_3s_prop'],color='#F59090')
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_5s_prop'],label=5,
       bottom=df_all_matches_bat_team_balltypes['bat_innings_0s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_1s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_2s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_3s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_4s_prop'],color='#0B0A0A')
ax.bar(df_all_matches_bat_team_balltypes['batting_team'],df_all_matches_bat_team_balltypes['bat_innings_6s_prop'],label=6,
       bottom=df_all_matches_bat_team_balltypes['bat_innings_0s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_1s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_2s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_3s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_4s_prop']+
           df_all_matches_bat_team_balltypes['bat_innings_5s_prop'],color='#EB2121')
plt.xticks(rotation=90)
bar0_legend = mpatches.Patch(color='#C5C5F7',label='0')
bar1_legend = mpatches.Patch(color='#8B8BEF',label='1')
bar2_legend = mpatches.Patch(color='#5151E7',label='2')
bar3_legend = mpatches.Patch(color='#EBBF12',label='3')
bar4_legend = mpatches.Patch(color='#F59090',label='4')
bar5_legend = mpatches.Patch(color='#0B0A0A',label='5')
bar6_legend = mpatches.Patch(color='#EB2121',label='6')
plt.legend(handles=[bar0_legend,bar1_legend,bar2_legend,bar3_legend,bar4_legend,bar5_legend,bar6_legend],
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../Plots/bar_stacked_runtype_teams.png")

# - Some teams (MI) face more dots but hit more 6s
# - Some teams (SH) face fewer dots and run more 2s but hit fewer 6s
# - More than anything, shows we can remove 3s and 5s from clustering

# Stacked bar plot by order as well (https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas)
df_all_matches_bat_team_balltypes_order = (df_all_matches_bat_inns.groupby(['batting_team','bat_order_striker_cat'], 
                                                                           as_index=False)['bat_innings_0s',
    'bat_innings_1s','bat_innings_2s','bat_innings_3s','bat_innings_4s','bat_innings_5s','bat_innings_6s',
    'bat_innings_balls_faced'].sum())

for i in list(range(0,7)):
        # as a proportion of balls faced
        df_all_matches_bat_team_balltypes_order['bat_innings_{0}s_prop'.format(i)] = (
                df_all_matches_bat_team_balltypes_order['bat_innings_{0}s'.format(i)]/
                df_all_matches_bat_team_balltypes_order['bat_innings_balls_faced'])

df_all_matches_bat_team_balltypes_order['bat_order_striker_cat_no'] = np.where(df_all_matches_bat_team_balltypes_order[
        'bat_order_striker_cat']=='Top',1,
        np.where(df_all_matches_bat_team_balltypes_order['bat_order_striker_cat']=='Middle',2,
                 np.where(df_all_matches_bat_team_balltypes_order['bat_order_striker_cat']=='Lower',3,4)))

df_all_matches_bat_team_balltypes_order_melt = pd.melt(df_all_matches_bat_team_balltypes_order, id_vars=['batting_team',
                                                                                                'bat_order_striker_cat'],
    value_vars=['bat_innings_0s_prop','bat_innings_1s_prop','bat_innings_2s_prop','bat_innings_3s_prop',
                'bat_innings_4s_prop','bat_innings_5s_prop','bat_innings_6s_prop'])

df_all_matches_bat_team_balltypes_order_melt['cumulative_value'] = (df_all_matches_bat_team_balltypes_order_melt.groupby(['batting_team','bat_order_striker_cat']).cumsum())

ball_colours = ['#C5C5F7','#8B8BEF','#5151E7','#EBBF12','#F59090','#0B0A0A','#EB2121']
fig,ax = plt.subplots(figsize=(10,12))     
for i, g in enumerate(df_all_matches_bat_team_balltypes_order_melt.groupby('variable')):
    ax = sns.barplot(data=g[1],
                     x='batting_team',
                     y='cumulative_value',
                     hue='bat_order_striker_cat',
                     color=ball_colours[i],
                     zorder=-i,
                     edgecolor='k')
plt.xticks(rotation=90)
ax.legend_.remove()
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("../Plots/bar_stacked_runtype_teams_order.png")

# - Top order of all teams face higher proportion of dots than middle order (powerplays are the obvious reason for this)
# - Some teams (SH) have smaller difference than others (MI) between top and middle for dots
# - Most teams' (CSK, DC, KKR, MI, RR, SH) lower orders hit greater proportion of balls for 6 than top and middle
# - Some top orders (PK) hit a lot more boundaries as a proportion of balls faced than others (SH)

# Try k-means clustering

numeric_features_1 = ['bat_innings_runs','bat_innings_balls_faced','bat_innings_0s_prop','bat_innings_1s_prop',
                      'bat_innings_2s_prop','bat_innings_4s_prop','bat_innings_6s_prop','bat_order_striker']
categorical_features_1 = ['bat_order_striker_cat']
prof_features_1 = ['bat_innings_runs','bat_innings_balls_faced','bat_innings_strike_rate','bat_innings_0s',
                   'bat_innings_0s_prop','bat_innings_1s','bat_innings_1s_prop','bat_innings_2s','bat_innings_2s_prop',
                   'bat_innings_4s','bat_innings_4s_prop','bat_innings_6s', 'bat_innings_6s_prop', 'bat_order_striker',
                   'bat_order_striker_cat_Top','bat_order_striker_cat_Middle','bat_order_striker_cat_Lower',
                   'bat_order_striker_cat_Tail','batting_team_Chennai Super Kings','batting_team_Delhi Capitals',
                   'batting_team_Kolkata Knight Riders','batting_team_Mumbai Indians','batting_team_Punjab Kings',
                   'batting_team_Rajasthan Royals','batting_team_Royal Challengers Bangalore',
                   'batting_team_Sunrisers Hyderabad']

df_clustered_1, GI_1 = clusinns_kmeans(df_all_matches_bat_inns,categorical_features_1,numeric_features_1,prof_features_1)

GI_1.to_csv("GI_1.csv")
pd.value_counts(df_clustered_1.inns_cluster)

df_clustered_2, GI_2 = clusinns_kmeans(df_all_matches_bat_inns,categorical_features_1,numeric_features_1,prof_features_1,
                                       max_nclus=40,n_init=20,max_iter=600,random_state=1)

GI_2.to_csv("GI_2.csv")
pd.value_counts(df_clustered_2.inns_cluster)

# - Clusters 1 (13 off 7), 4 (61 off 42) and 5 (21 off 9) could broadly be characterised as 'good innings'
# - Cluster 1 is also characterised by being played by lower order batters and tailenders and over-indexing on 4s. It is
#   played mostly by DC and KKR
# - Cluster 4 over-indexes on every type of scoring shot, but less so on 4s than the rest, and under-indexes on dots faced.
#   80% of these innings are played by top order batters and the teams that play them most are SH and CSK.
# - Cluster 5 over-indexes on 6s and is played mostly by middle and lower order batters. The teams that play them most are
#   KKR and MI.
# - Cluster 0 (14 off 14) looks like an attempt from a top order batter to bat long that has failed. Over-indexes on dots
#   and 4s. RR over-index on this slightly more than the other teams.
# - Cluster 3 (27 off 21) could be a decent middle-order innings, depending on the situation. Under-indexed on dots, over-
#   on 1s, 2s and 6s. CSK over-index slightly more than the other teams.
# - Cluster 2 (1 off 3) is a short innings by a tailender.CSK and SH are particularly good at avoiding these.
# - Cluster 6 (8 off 7) is a hard-running innings played by a lower-order batter or tailender. Over-indexed on 2s, under-
#   on all other balls, including dots. CSK, PK, SH.
# - Cluster 7 (4 off 4) is similar to cluster 6 in terms of who plays the innings but over-indexes on 1s. PK, SH.
# - Cluster 8 (6 off 8) is an innings played by middle- and lower-order batters and tailenders where they fail to get going.
#   Over-indexed on dots and 1s. PK, RCB.

# Box/violin plots of runs scored and balls faced by cluster

fig,ax = plt.subplots(3,3, figsize=(10,12))
sns.boxplot(ax=ax[0,0],data=df_clustered_2,x='inns_cluster',y='bat_innings_runs')
sns.boxplot(ax=ax[0,1],data=df_clustered_2,x='inns_cluster',y='bat_innings_balls_faced')
sns.boxplot(ax=ax[0,2],data=df_clustered_2,x='inns_cluster',y='bat_innings_strike_rate')
sns.violinplot(ax=ax[1,0],data=df_clustered_2,x='inns_cluster',y='bat_order_striker')
sns.boxplot(ax=ax[1,1],data=df_clustered_2,x='inns_cluster',y='bat_innings_0s_prop')
sns.boxplot(ax=ax[1,2],data=df_clustered_2,x='inns_cluster',y='bat_innings_1s_prop')
sns.boxplot(ax=ax[2,0],data=df_clustered_2,x='inns_cluster',y='bat_innings_2s_prop')
sns.boxplot(ax=ax[2,1],data=df_clustered_2,x='inns_cluster',y='bat_innings_4s_prop')
sns.boxplot(ax=ax[2,2],data=df_clustered_2,x='inns_cluster',y='bat_innings_6s_prop')
ax[0,0].set(ylabel='Runs scored',xlabel='Innings cluster')
ax[0,1].set(ylabel='Balls faced',xlabel='Innings cluster')
ax[0,2].set(ylabel='Strike rate',xlabel='Innings cluster')
ax[1,0].set(ylabel='Batting order',xlabel='Innings cluster')
ax[1,1].set(ylabel='0s proportion',xlabel='Innings cluster')
ax[1,2].set(ylabel='1s proportion',xlabel='Innings cluster')
ax[2,0].set(ylabel='2s proportion',xlabel='Innings cluster')
ax[2,1].set(ylabel='4s proportion',xlabel='Innings cluster')
ax[2,2].set(ylabel='6s proportion',xlabel='Innings cluster')
plt.tight_layout()
plt.savefig("box_1.png")

# See if I can increase the number of meaningful clusters by increasing max_nclus, n_init and max_iter. Remove absolute
# number of 0s, 1s, ..., 6s while I'm at it.

prof_features_3 = ['bat_innings_runs','bat_innings_balls_faced','bat_innings_strike_rate','bat_innings_0s_prop',
                   'bat_innings_1s_prop','bat_innings_2s_prop','bat_innings_4s_prop','bat_innings_6s_prop',
                   'bat_order_striker','bat_order_striker_cat_Top','bat_order_striker_cat_Middle',
                   'bat_order_striker_cat_Lower','bat_order_striker_cat_Tail','batting_team_Chennai Super Kings',
                   'batting_team_Delhi Capitals','batting_team_Kolkata Knight Riders','batting_team_Mumbai Indians',
                   'batting_team_Punjab Kings','batting_team_Rajasthan Royals','batting_team_Royal Challengers Bangalore',
                   'batting_team_Sunrisers Hyderabad']

df_clustered_3, GI_3 = clusinns_kmeans(df_all_matches_bat_inns,categorical_features_1,numeric_features_1,prof_features_3,
                                       max_nclus=60,n_init=40,max_iter=1000,random_state=1)
GI_3.to_csv('GI_3.csv')

# This looks even better at first glance

# Add absolute number of 0s, 1s, ..., 6s to see if it increases ability to tease out other clusters.

numeric_features_4 = ['bat_innings_runs','bat_innings_balls_faced','bat_innings_0s_prop','bat_innings_1s_prop',
                      'bat_innings_2s_prop','bat_innings_4s_prop','bat_innings_6s_prop','bat_order_striker',
                      'bat_innings_0s','bat_innings_1s','bat_innings_2s','bat_innings_4s','bat_innings_6s']

df_clustered_4, GI_4 = clusinns_kmeans(df_all_matches_bat_inns,categorical_features_1,numeric_features_4,prof_features_3,
                                       max_nclus=60,n_init=40,max_iter=1000,random_state=1)
GI_4.to_csv('GI_4.csv')
# This looks pretty good as well. Some differentiation between two types of 'good' top order innings











# Line plots to show different shapes of innings

df_all_matches['bat_inns_id'] = df_all_matches.groupby(['match_id','striker']).ngroup().astype(str).str.zfill(5)

sns.lineplot(data=df_all_matches,x='bat_innings_balls_faced',y='bat_innings_runs',hue='bat_inns_id')
# This takes ages, will have to rethink



