"""
Here lives our movie recommenders functions
"""
import pandas as pd
import numpy as np
import random
from utils import movies ,nmf_model,movie_1
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity


def nmf_recommender(query:dict,model=nmf_model,k=10)->list:
    """_summary_

    Args:
        quer (dict): _description_
        model (_type_): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    Q =nmf_model.components_
    # 1. candidate generation
    data = list(query.values())   # the ratings of the new user
    row_ind = [0]*len(data)       # we use just a single row 0 for this user 
    col_ind = list(query.keys())
    
    # construct a user vector
    R_user = csr_matrix((data, (row_ind, col_ind)), shape = (1, Q.shape[1]))
    P_user = model.transform(R_user)
    # Q does not change
    R_recommended = np.dot(P_user, Q)
   
    # 2. scoring
    # convert to a pandas series
    scores = pd.Series(R_recommended[0])
    
    # give a zero score to movies the user has already seen
    scores[query.keys()] = 0
    
    # 3. ranking
    # sort the scores from high to low 
    scores = scores.sort_values(ascending=False)
    

    # filter out movies allready seen by the user
    # get the movieIds of the top 10 entries
    recommendations = scores.head(10).index
    
    # return the top-k highst rated movie ids or titles
    movies = pd.read_csv('./movies.csv')
    top10=movies.set_index('movieId').loc[recommendations]['title']
    return top10

def recommend_cosin(query,k=10):
    df=pd.read_csv("./user_movie_data.csv")
    df=df.fillna(value=0)
    df.set_index('userId',inplace=True)
    
    # initialize new user 
    new_user=np.zeros_like(df.columns)
    #(To go back to an array if you have a dictionary query)
    
    for index,item in enumerate(df.columns):
        if query.get(item):# <-- Return the value for key if key is in the dictionary
            #change the rating by input data
            new_user[index]=query[item]
    
    # new user dataframe
    new=pd.DataFrame([new_user],index=[len(df)+1],columns=df.columns)
    
    #add new user to df dataframe
    df=pd.concat([df,new],ignore_index=True)
    # We can turn this into a dataframe:
    cosine_sim_table2 = pd.DataFrame(cosine_similarity(df), index=df.index, columns=df.index)
    df2_t=df.T
    # choose an active user
    active_user = len(df)-1
    # create a list of unseen movies for this user
    unseen_movies = list(df2_t.index[df2_t[active_user] == 0])
    # Create a list of top 3 similar user (nearest neighbours)
    neighbours = list(cosine_sim_table2[active_user].sort_values(ascending=False).index[1:4])
    # create the recommendation (predicted/rated movie)
    predicted_ratings_movies = []

    for movie in unseen_movies:
    
        # we check the users who watched the movie
        people_who_have_seen_the_movie = list(df2_t.columns[df2_t.loc[movie] > 0])
    
        num = 0
        den = 0
        for user in neighbours:
            # if this person has seen the movie
            if user in people_who_have_seen_the_movie:
                #  we want extract the ratings and similarities
                rating = df2_t.loc[movie, user]
                similarity = cosine_sim_table2.loc[active_user, user]
            
                # predict the rating based on the (weighted) average ratings of the neighbours
                # sum(ratings)/no.users OR 
                # sum(ratings*similarity)/sum(similarities)
                num = num + rating*similarity
                den = den + similarity
        if den != 0:
            predicted_ratings = num/den
        else:
            predicted_ratings = 0
        predicted_ratings_movies.append([predicted_ratings,movie])
        # create df pred
    df_pred = pd.DataFrame(predicted_ratings_movies,columns = ['rating','movie'])
    recommendation=df_pred.sort_values(by=['rating'],ascending=False)['movie'].head(k)
    recommendation=recommendation.tolist()
    return recommendation   




if __name__=="__main__":
    top10 = nmf_recommender(movies)
    cosin_result=recommend_cosin(movie_1)
    print("nmf_result\n",top10)
    print("cosimalirity_result\n",cosin_result)

