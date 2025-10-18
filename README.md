# Import required libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Step 1: Sample user-item rating data
data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Item': ['Item1', 'Item2', 'Item3', 'Item2', 'Item3', 'Item1', 'Item3', 'Item1', 'Item2'],
    'Rating': [5, 3, 4, 4, 5, 2, 4, 4, 3]
}

df = pd.DataFrame(data)

# Step 2: Create user-item matrix
user_item_matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)

# Step 3: Compute item-item similarity matrix
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Step 4: Recommend items for a user
def recommend_items(user_name, user_item_matrix, item_similarity_df, top_n=2):
    user_ratings = user_item_matrix.loc[user_name]
    scores = user_ratings.dot(item_similarity_df) / item_similarity_df.sum(axis=1)
    scores = scores.sort_values(ascending=False)
    
    # Filter out items already rated
    recommended = scores[~user_ratings.index.isin(user_ratings[user_ratings > 0].index)]
    return recommended.head(top_n)

# Step 5: Show recommendations for User 'C'
recommendations = recommend_items('C', user_item_matrix, item_similarity_df)
print("Recommended items for User C:")
print(recommendations)
