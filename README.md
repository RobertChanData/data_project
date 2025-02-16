# Spotify Project

### Project Overview
This project is part of my Master's program: "Machine Learning I".

Spotify's music streaming service allows users to create personalized playlists and manage family accounts.  
Due to a server hack, data regarding `user` and `top_year` for some songs were deleted.  
The task is to:
 - Reconstruct the Top-of-the-Year playlists for each user from 2018 to 2024 using the available data
 - Reconstruct the missing user: Recover the missing songs for each user’s playlist

### Project summary

The Jupyter Notebook is structured in 5 ways:
1. Exploratory Data Analysis (EDA)
2. Data Preparation (preprocessing)
3. User Playlist Reconstruction using Machine Learning
4. Top-Year Playlist Reconstruction using Machine Learning
5. Reconstruction the Mlaylist Data

### Key Result

**User playlist** was reconstructed using **OneVSRestClassifier** (Logisitc Regression) with F1-score of **87%**.  
- 5 class to predict
- Before feature enigneering, model performance was at best 40% using more complex model (SVC, Random Forest, OvR with degree-3 poly kernel)

**Top-year playlist** was reconstructed using **Decision Tree Classifier** with F1-score of **76%**.



### 1. Exploratory Data Analysis EDA
