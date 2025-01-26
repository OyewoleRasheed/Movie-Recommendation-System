#This system will recommend movies based on user-user relationship and title of movie, creating a hybrid recommendation system combining both collaborative and content based movie recommendation

#%%
#Import libriaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
#%%
#Create column names
col_name = ['userid','itemid','rating','timestamp']
#Load data
data = pd.read_csv(r'Data\u.data', sep ='\t', names = col_name)
#Turn the data into a dataframe
df = pd.DataFrame(data)
#%%
#Explore dataset
#View the first five rows
df.head()
#Count the number of rows in the dataset
df.count()
#Check for null values
df.isna().sum()
#Look at the data structure
df.info()
#%%
#Extract date from timestamp
df["date"] = pd.to_datetime(df["timestamp"], unit='s')
#Extract year from date
df["Year"] = df["date"].dt.year

df.head()
#%%
#Load movie title dataset
mov_data = pd.read_csv(r'Data\Movie_Id_Titles.txt')
#%%
#Explore the dataset
mov_data.head()
mov_data.info()
mov_data.rename(columns={"item_id":"itemid"}, inplace = True)

#%%
#Merge movie ratings with movie title

Movie= pd.merge(df,mov_data,on='itemid')
Movie.isna().sum()
Movie.head()
#Reduce movie to 1000 due to computational resources
new_Movie = Movie.head(1000)


#%%
#Check the structure of the new_Movie dataset
new_Movie.info()

#%%
#Group movies by title , find the average of rating and arrange movie title by average ratings in descending order
Sorted_Movie= new_Movie.groupby('title')['rating'].mean().sort_values(ascending = False)
Movie_ratings = pd.DataFrame(Sorted_Movie)
Movie_ratings.head()
#%%
Movie_ratings['Ratings Count'] = new_Movie.groupby('title').rating.count()
rating_count = new_Movie.groupby('title')['rating'].count()
Movie_ratings.reset_index()
Movie_ratings = Movie_ratings.merge(new_Movie[['title', 'Year']].drop_duplicates(), on='title', how='left')

#%%%
#Create a user item matrix
user_item_matrix = new_Movie.pivot_table(index = "userid",columns= "title",values ="rating").fillna(0)
#%%
svd = TruncatedSVD(n_components = 20, random_state = 42)
matrix_svd = svd.fit_transform(user_item_matrix)

#%%
#calculate user to user similarity
user_sim = cosine_similarity(matrix_svd)  

#%%
#User similarity dataframe
user_sim_df = pd.DataFrame(user_sim, index =user_item_matrix.index, columns = user_item_matrix.index)

#%%
#function to get recommendations based on similar users
def collaborative_recommendations(user_id, top_n=5):
    #find similar users
    similar_users = user_sim_df[user_id].sort_values(ascending = False).iloc[1:top_n +1]
    #Get movie prescence of similar users
    similar_users_rating = user_item_matrix.loc[similar_users.index].mean(axis = 0)

    recommendations = similar_users_rating.sort_values(ascending = True).head(top_n)

    return recommendations
#%%
#For content based movie recommendations
# Apply TF-IDF on movie titles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_Movie['title'])
#%%
# Compute cosine similarity between movies
tfidf_matrix_sparse = csr_matrix(tfidf_matrix)
movie_similarity_sparse = cosine_similarity(tfidf_matrix_sparse, dense_output=False)


#%%

# Create a DataFrame for movie similarity (sparse to dense for specific queries)
movie_sim_df = pd.DataFrame.sparse.from_spmatrix(
    movie_similarity_sparse, 
    index=new_Movie['title'], 
    columns=new_Movie['title']
)

movie_sim_df
#%%
def content_based_recommendations(movie_title, top_n=5):
    # Get similarity scores for the given movie
    similar_movies = movie_sim_df[movie_title].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_movies


#%% 
# Hybrid Recommendation Function
def hybrid_recommendations(user_id, movie_title, top_n=5, alpha=0.5):
    """
    Combine collaborative and content-based recommendations.
    :param user_id: User ID for collaborative filtering
    :param movie_title: Movie title for content-based filtering
    :param top_n: Number of recommendations to return
    :param alpha: Weight for collaborative (1-alpha for content-based)
    """
    # Collaborative recommendations
    collab_recs = collaborative_recommendations(user_id, top_n=top_n)
    
    # Content-based recommendations
    content_recs = content_based_recommendations(movie_title, top_n=top_n)
   

    #Checking for duplicates

    if collab_recs.index.duplicated().any():
        collab_recs = collab_recs.groupby(collab_recs.index).mean()

    if content_recs.index.duplicated().any():
        content_recs = content_recs.groupby(content_recs.index).mean()
    # Merge recommendations
    hybrid_scores = pd.concat([
        collab_recs.rename("Collaborative Score"),
        content_recs.rename("Content-Based Score")
    ], axis=1).fillna(0)
    
   
    # Weighted average
    hybrid_scores['Hybrid Score'] = alpha * hybrid_scores['Collaborative Score'] + (1-alpha) * hybrid_scores['Content-Based Score']
    
    return hybrid_scores.sort_values('Hybrid Score', ascending=False).head(top_n)


#%%

# Test the recommender
user_id = 200

movie_title = "Heavyweights (1994)"  # Replace with a valid title from your dataset


#%%
recommendations = hybrid_recommendations(user_id, movie_title, top_n=5, alpha=0.7)
#%%
print(recommendations)


#%%
plt.figure(figsize = (12,10))
Movie_ratings["Ratings Count"].hist(bins=30)
plt.ylabel("Frequency of Rating Counts")
plt.ylabel("Rating Counts")
plt.title("Histogram of Ratings Counts")
plt.show()
#%%
plt.figure(figsize=(12,8))
rating_count.head().sort_values(ascending=True).plot(kind='barh')

plt.show()
# %%
plt.figure(figsize=(12,12))
rating_count.tail(15).sort_values(ascending=True).plot(kind='barh')

plt.show()
# %%
rating_count.sort_values().head(15)

# %%
Movie_ratings
# %%
plt.figure(figsize=(12,8))
plt.scatter(Movie_ratings["Ratings Count"],
Movie_ratings["rating"], alpha= 0.7, color ="coral")
plt.title("Relationship Between Number of Ratings and Average Rating")
plt.xlabel("Number of Ratings (Rating Count)")
plt.ylabel("Average Rating")
plt.grid(alpha=0.5)
plt.tight_layout()
# %%

Movie_ratings["Popularity"]= pd.cut(Movie_ratings["Ratings Count"], bins=[0, 200,400, 500,Movie_ratings["Ratings Count"].max()], labels=["low","Medium","High","Very High"])

popularity_ratings = Movie_ratings.groupby("Popularity")["rating"].mean()

popularity_ratings.plot(kind="bar", 
                        color = "Blue",
                        legend=False)
plt.title("Average Rating by Popularity Category")
plt.xlabel("Popularity Category")
plt.ylabel("Average Rating")
plt.grid(axis= "y", linestyle = "--", alpha = 0.5)
plt.tight_layout()


#%%
ratings_by_year= Movie_ratings.pivot_table(
    index= "Year",
    values = "Ratings Count",
    aggfunc="sum"
)
plt.figure(figsize=(12, 6))
sns.heatmap(ratings_by_year, cmap="Reds", annot =True,fmt=".0f", cbar= False)
# sns.heatmap(ratings_by_year, cmap="Blues", annot=True, fmt=".0f", cbar=False)
plt.title("Ratings Count by Year")
plt.xlabel("Year")
plt.ylabel("Ratings Count")
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x="Popularity", y="rating", data=Movie_ratings,palette="Set3")
plt.title("Spread of Ratings by Popularity Category")
plt.xlabel("Popularity Category")
plt.ylabel("Average Rating")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
from wordcloud import WordCloud
Movie_title_wordcloud =WordCloud(background_color = "White", width = 800, height = 400).generate_from_frequencies(Movie_ratings.set_index("title")["Ratings Count"])

plt.figure(figsize=(12, 8))
plt.imshow(Movie_title_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Movie Titles (Sized by Rating Counts)")
plt.show()
# %%
#