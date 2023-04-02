import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def encode_all_nba(df, all_nba_players, season):
    '''
    Encodes a 1 if a player made the All-NBA team, 0 if not
    
    df: dataframe which you want to add new encoded column
    all_nba_players: list of player names who made an All-NBA team
    season: season you are encoding for 
    
    returns data frame with added column
    '''
    season_df = df[df['season'] == season].copy()
    season_df['All-Nba'] = 0
    for player in all_nba_players:
        if player in season_df['Player'].values:
            # find the row index for the player
            row_idx = season_df[season_df['Player'] == player].index[0]
            # update the "All-NBA" column for the player to 1
            season_df.at[row_idx, 'All-Nba'] = 1
        else:
            print(f"{player} not found in DataFrame.", season)
    df[df['season'] == season] = season_df
    return df


def check_all_nba(df):
    '''
    prints total number of All-NBA players using sum, to ensure binary encoding is correct and all the player names with 1 
    '''
    x = df[df['All-Nba'] == 1]
    print("Total all nba players: ", len(x))
    print(x[['Player', 'season']])
    print(x.columns)
    

def pos_split(df):
    '''
    splits data frame into guards and front court players (forwards,centers)

   '''
    #df.insert(0, 'index', df.index)
    guards = ["SG","PG"]
    fronts = ["SF","PF"]
    cents = ["C"]
    
    df_guards = df[df["Pos"].isin(guards)]
    df_fronts = df[df["Pos"].isin(fronts)]
    df_cents = df[df["Pos"].isin(cents)]
    
    df_guards = df_guards.drop(["Pos"], axis=1)
    df_fronts = df_fronts.drop(["Pos"], axis=1)
    df_cents = df_cents.drop(["Pos"], axis=1)

       
    return df_guards,df_fronts,df_cents

def normalize(df):
    '''
    normalizes feature vector and drops correct columns for X and y
    
    '''
    X = df.drop(['Player',"season", 'All-Nba'], axis=1)
    y = df['All-Nba']
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    return X, y



def top_score_df(testdf,y_pred,topnum):
    '''
    creates a dataframe containing model predictions based on top specified prediction scores
    '''
    pred_df = pd.DataFrame()
    pred_df["Player"] = testdf["Player"]
    pred_df["Ground Truth All-NBA Score"] = testdf["All-Nba"]
    pred_df["pred score"] = y_pred

    ypredsort = sorted(pred_df["pred score"], reverse=True)[:topnum]
    pred_df["predictions"] = 0
    pred_df["predictions"][pred_df["pred score"].isin(ypredsort)] = 1
    pred_df["season"] = testdf["season"]
    return pred_df


def top_score_df2(testdf,y_pred,topnum):
    '''
    creates a dataframe containing model predictions based on top specified prediction scores
    '''
    pred_df = pd.DataFrame()
    pred_df["Player"] = testdf["Player"]
    pred_df["pred score"] = y_pred

    ypredsort = sorted(pred_df["pred score"], reverse=True)[:topnum]
    pred_df["predictions"] = 0
    pred_df["predictions"][pred_df["pred score"].isin(ypredsort)] = 1
    pred_df["season"] = testdf["season"]
    return pred_df

def thres_score_df(testdf,y_pred,thres):
    '''
    creates a dataframe containing model predictions based on which prediction scores are above a specifc threshold
    '''
    
    pred_df = pd.DataFrame()
    #pred_df  = y_test.to_frame().reset_index()
    pred_df["Player"] = testdf["Player"]
    pred_df["Ground Truth All-NBA Score"] = testdf["All-Nba"]
    pred_df["pred score"] = y_pred
    pred_df["season"] = testdf["season"]

    
    pred_df["predictions"] = 0
    pred_df["predictions"][pred_df["pred score"] > thres] = 1
    pred_df["season"] = testdf["season"]
    return pred_df


def all_image(pred_df,color):
    x = list(pred_df["correct predictions"])
    imshowpred = np.reshape(x, (len(x), 1))
    plt.imshow(imshowpred, cmap=color, aspect='auto')
    plt.show()
    
    
def print_all_pred(pred_df):
    print("Number of correct predictions:",(np.sum(pred_df["correct predictions"])) , "out of",len(pred_df["correct predictions"])) 
    print("Percentage of correct predictions:",(np.sum(pred_df["correct predictions"])) / (len(pred_df["correct predictions"])))
    
    
    
def all_nba_image(pred_df,color):
    x = list(pred_df["correct predictions"][pred_df["Ground Truth All-NBA Score"] == 1])
    imshowpred = np.reshape(x, (len(x), 1))
    plt.imshow(imshowpred, cmap=color, aspect='auto')
    plt.show()

    
    
def print_all_nba_pred(pred_df):
    print("Number of correct predictions:",(np.sum(pred_df["correct predictions"][pred_df["Ground Truth All-NBA Score"] == 1])) , "out of",len(pred_df["correct predictions"][pred_df["Ground Truth All-NBA Score"] == 1])) 
    print("Percentage of correct predictions:",(np.sum(pred_df["correct predictions"][pred_df["Ground Truth All-NBA Score"] == 1])) / (len(pred_df["correct predictions"][pred_df["Ground Truth All-NBA Score"] == 1])))

    
def missed_pred(pred_df):
    '''
    Gets players who were not predicted to be on an All-NBA team when they were
    '''
    x1 = pred_df[pred_df["Ground Truth All-NBA Score"] == 1]
    return x1[x1["correct predictions"] == False].reset_index()



def false_pos(pred_df):
    '''
    Gets players who were predicted to be on an All-NBA team but actually were not
    '''
    x1 = pred_df[pred_df["correct predictions"] == False]
    return x1[x1["predictions"] == 1].reset_index()




def means_df(testdf2,pred_df):
    '''
    compares means of false positive and negative
    '''
    false_negative_means = testdf2.iloc[false_pos(pred_df)["index"]].mean()
    z = pd.DataFrame(false_negative_means)
    
    false_positive_means = testdf2.iloc[missed_pred(pred_df)["index"]].mean()
    z1 = pd.DataFrame(false_positive_means)
    
    false_means = pd.concat([z,z1],axis = 1)
    false_means.columns = ["False Negative Means", "False Positive Means"]
    return false_means



def overlap_bar(meansdf):
    trace1 = go.Bar(x=meansdf.index, y=meansdf['False Negative Means'], name='False Negative Means')
    trace2 = go.Bar(x=meansdf.index, y=meansdf['False Positive Means'], name='False Positive Means')

    # create the layout
    layout = go.Layout(title='Differences in False Positive and Negative', barmode='group')

    # create the figure
    fig = go.Figure(data=[trace2, trace1], layout=layout)

    # plot the figure
    fig.show()


