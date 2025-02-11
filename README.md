﻿# Movie-Recommendation-System
#This project is a hybrid recommendation system that combines collaborative filtering and content-based filtering to recommend movies to users. The system leverages user-movie interactions and movie metadata to create personalized and relevant recommendations.

## Features

- Collaborative Filtering:
  - Uses Singular Value Decomposition (SVD) to compress the user-item matrix.
  - Calculates cosine similarity between users to recommend movies based on user-user relationships.
  
- Content-Based Filtering:
  - Uses TF-IDF to compute similarity between movie titles.
  - Recommends movies similar to a specified title based on cosine similarity.

- Hybrid Recommendations:
  - Combines collaborative and content-based scores using a weighted average approach.

- Visualization:
  - Plots histograms, scatterplots, and heatmaps to analyze movie ratings and trends.
  - Includes a word cloud to visualize movie titles by popularity.

---

# Dataset

- MovieLens Dataset: Contains information about users, movies, ratings, and timestamps.
  - User-movie interactions: `u.data`
  - Movie metadata: `Movie_Id_Titles.txt`
- The dataset is preprocessed to include columns like movie titles, years, and user-item ratings.

- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `wordcloud`


## Future Improvements

- Incorporate more advanced collaborative filtering models (e.g., matrix factorization with neural networks).
- Use additional movie metadata (e.g., genres, actors) for better content-based filtering.
- Optimize the hybrid approach by dynamically adjusting weights.
