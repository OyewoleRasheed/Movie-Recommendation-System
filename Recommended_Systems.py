#%%
#Import libriaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
#Name columns
col_name = ['userid','itemid','rating','timestamp']
#Load data
data = pd.read_csv(r'Data\u.data', sep ='\t', names = col_name)
#Turn the data into a dataframe
df = pd.DataFrame(data)
#%%
#View the first five rows
df.head()
#Count the number of rows in the dataset
df.count()
#Check for null values
df.isna().sum()
#Look at the data structure
df.info()
#%%
df["date"] = pd.to_datetime(df["timestamp"], unit='s')

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
#%%
#Exploratory Data Analysis

Sorted_Movie= Movie.groupby('title')['rating'].mean().sort_values(ascending = False)
Movie_ratings = pd.DataFrame(Sorted_Movie)
Movie_ratings.head()
#%%
Movie_ratings['Ratings Count'] = Movie.groupby('title').rating.count()
rating_count = Movie.groupby('title')['rating'].count()
Movie_ratings.reset_index()
Movie_ratings = Movie_ratings.merge(Movie[['title', 'Year']].drop_duplicates(), on='title', how='left')

Movie_ratings.head()

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
#%%
most_rated_movies = Movie_ratings.sort_values(by='Ratings Count', ascending= False).head(10)
most_rated_movies= most_rated_movies.reset_index()
most_rated_movies.plot(
    x="title",
    y="Ratings Count",
      kind="barh",
      color= "purple",
      legend= False)
plt.title("Top 15 Movies with the highest number of Ratings")
plt.ylabel("Movie Title")
plt.xlabel("Number of Ratings")
plt.gca().invert_yaxis()

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
#new