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

### Exploratory from the Dataset
We are trying to predict the missing `user` and `top_year` (far-left of the dataframe)  
The first row is missing the user and top_year that was deleted and that we need to predict

We have 3 mains features: 
- metadata about the track: name, album, artist, duration (length)
- release date and top_year
- song characteristic like tempon energy, accousticness

Target to predict are the missing value in:
- user: {`alpha`, `beta`, `gamma`, `delta`, `epsilon`}
- top_year: {`2018`, `201`9, `2020`, `2021`, `2022`, `2023`, `2024`}

*__Table__: Sample of the dataset*
| name | album | artist | release_date | length | popularity | acousticness | danceability | energy | instrumentalness | liveness | loudness | speechiness | tempo | valence | time_signature | key | mode | uri | release_year | top_year | user |
|------|-------|--------|--------------|--------|------------|--------------|--------------|--------|------------------|---------|----------|-------------|-------|---------|----------------|-----|------|-----|--------------|----------|------|
| Variations on a Polish Theme, Op. 10: No. 5 Andantino | Szymanowski: Piano Works, Vol. 2 | Karol Szymanowski | 06/12/1996 | 76933 | 53 | 0.9960 | 0.329 | 0.00695 | 0.866000 | 0.0906 | -34.227 | 0.0448 | 70.295 | 0.238 | 4 | 11 | 0 | spotify:track:3bcdLMrAxrfn5dxInjIdI2 | 1996 | unknown | unknown |
| Je vous trouve un charme fou - En duo avec Gaëtan Roussel | Il suffit d'y croire (Version deluxe) | Hoshi | 2018-11-30 | 172626 | 62 | 0.6220 | 0.615 | 0.59900 | 0.000008 | 0.1920 | -8.715 | 0.2530 | 86.976 | 0.626 | 4 | 1 | 1 | spotify:track:0C2yaSWVgCUiiqPyYxSOkd | 2018 | 2024 | delta |
| Me Gusta | On ira où ? | DTF | 2019-10-11 | 175269 | 72 | 0.4130 | 0.834 | 0.73400 | 0.000040 | 0.1130 | -5.948 | 0.3410 | 89.989 | 0.356 | 4 | 6 | 0 | spotify:track:6P3FBaZfUjeWYExU2ShaPZ | 2019 | 2022 | gamma |
| L’amour en Solitaire | Petite Amie (Deluxe) | Juliette Armanet | 2018-02-02 | 175266 | 0 | 0.4040 | 0.797 | 0.50600 | 0.000153 | 0.2550 | -6.774 | 0.0327 | 128.027 | 0.539 | 4 | 5 | 0 | spotify:track:2tn51grfchxArwPXeXkoX5 | 2018 | 2020 | gamma |

### General data

- The dataset contains **3600** rows.  
- **2 duplicate entries** were identified and removed. 
- 9 songs with no name, album and artist = `Various artist`. After verification, it is not an input error.  
  Given their low occurrence, they were removed to maintain data quality without significantly impacting model performance.
- Predicted class are **perfectly balanced** (no bias from class imbalanced)

*__Contingency table__: between user and top_year*
| top_year | alpha | beta | delta | epsilon | gamma | unknown |
|----------|-------|------|-------|---------|-------|---------|
| 2018     | 100   | 100  | 100   | 100     | 100   | 0       |
| 2019     | 100   | 100  | 100   | 100     | 98    | 0       |
| 2020     | 100   | 100  | 100   | 95      | 100   | 0       |
| 2021     | 100   | 100  | 96    | 100     | 100   | 0       |
| 2022     | 100   | 100  | 100   | 100     | 100   | 0       |
| 2023     | 100   | 100  | 100   | 100     | 100   | 0       |
| 2024     | 100   | 100  | 100   | 100     | 100   | 0       |
| unknown  | 0     | 0    | 0     | 0       | 0     | 100     |

### Descriptive Analysis

Each track has distinct musical features that differentiate it from others.  
*Source: https://developer.spotify.com/documentation/web-api/reference/get-audio-features*

How about an real example to illustrate it?

🎵 Don't Stop Me Now (Remastered 2011) - Queen 🎵  
[Listen on Spotify](https://open.spotify.com/track/5T8EDUDqKcs6OSOwEsfqG7)  

| **Song**                    | **Album**            | **Artist** | **Release Date** | **Length**     | **Popularity** | **Acousticness** | **Danceability** | **Energy** | **Instrumentalness** | **Liveness** | **Loudness** | **Speechiness** | **Tempo** | **Valence** | **Key**  | **Top Year** | **User** |                                            |
|-----------------------------|----------------------|------------|------------------|----------------|----------------|------------------|------------------|------------|----------------------|--------------|--------------|-----------------|-----------|-------------|----------|--------------|----------|--------------------------------------------------|
| Don't Stop Me Now - Remastered 2011 | Jazz (2011 Remaster) | Queen      | 1978-11-10       | 3m 29s         | 85%            | 4.75%            | 55.9%            | 86.8%      | 0.02%                | 77.6%        | -5.28 dB     | 17%             | 156 BPM   | 60.9%       | F Major | 2020         | Alpha

| **Feature**         | **Value**     | **Description**                                  |
|---------------------|---------------|--------------------------------------------------|
| **Acousticness**     | 4.75%         | Mostly electric, with little acoustic presence   |
| **Danceability**     | 56%           | Energetic, catchy rhythm, dance-friendly         |
| **Energy**           | 86.8%         | High energy, powerful, perfect for a lively vibe |
| **Instrumentalness** | 0.02%         | Vocal-driven with minimal instrumental sections  |
| **Liveness**         | 77.6%         | Studio recording with live performance energy    |
| **Loudness**         | -5.28 dB      | Loud and dynamic, fitting for rock anthems       |
| **Speechiness**      | 17%           | Some spoken moments, mainly singing             |
| **Tempo**            | 156 BPM       | Fast-paced, adding to the upbeat vibe            |
| **Valence**          | 60.9%         | Positive, fun, and uplifting                    |
| **Key**              | F Major       | Bright and open, matching the energetic tone    |

Here is the boxplot for each column. *The complete analysis is in the notebook*

*__Boxplot__ of the differents musical features*
![Alt text](https://github.com/RobertChanData/spotify_project/blob/main/Screenshot/Spotify_1.PNG?raw=true)
