<h1 align="center">Spotify Project</h1>

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



<h2 align="center">  :star: Key Result  :star: </h2>

This **data imputation** problem was approached as a **classification problem**.  

**User playlist** was reconstructed using **OneVSRestClassifier** (Logistic Regression) with F1-score of **87%**.  
- **5 class** to predict: {`alpha`, `beta`, `gamma`, `delta`, `epsilon`}
- Before feature engineering, model performance was at best **40%** using more complex model (SVC, Random Forest, OvR with degree-3 poly kernel)
- To prevent over-reliance on the newly feature created, **L2 penalty** (Ridge penalization) was added. It reduce overfitting and give more room for other features

**Top-year playlist** was reconstructed using **Decision Tree Classifier** with F1-score of **76%**.
- **7 class** to predict: {`2018`, `201`9, `2020`, `2021`, `2022`, `2023`, `2024`}
- Before feature engineering, model performance was at best **40%** using more complex model (k-NN, Random Forest Classifier, Gradient Boosting)
- To prevent overfitting, **hyperparameter tuning** using grid was performed and 

**Output**: The dataset with the reconstructed user and top_year predicted by both models.  

<h1 align="center">Exploratory Data Analysis EDA</h1>

The Exploratory Data Analysis is structured in 9 sub-sections:

**1.** Data Exploration - **2.** Handling missing values - **3.** Check for duplicate - **4.** Explore the Target variable - **5.** Data Type - **6.** Descriptive Analysis - **7.** Correlation Analysis - **8.** Feature-Target Relationship Analysis (User) - **9.** Feature-Target Relationship Analysis (Top_year)
