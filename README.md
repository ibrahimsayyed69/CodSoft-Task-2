# Movie Rating Prediction using Machine Learning 🎬🍿
This repository contains a Data Science project focused on predicting the rating of a movie based on its characteristics (such as genre, director, actors, budget, or release year). This is a regression task where we analyze how different features influence audience or critic reception.
## 📌 Project Overview
The objective of this project is to explore historical movie datasets, perform detailed exploratory data analysis (EDA), and build a predictive regression model. By analyzing factors like the cast, director, runtime, and release timeline, the model aims to estimate the historical rating score a movie receives.
## 📊 Dataset Description
The dataset typically includes historical data of movies with features such as:
* `Name` / `Title`: The title of the movie.
* `Year`: The year the movie was released.
* `Duration`: The total runtime of the movie in minutes.
* `Genre`: The categories the movie falls into (e.g., Action, Drama, Comedy).
* `Rating`: The target continuous variable representing the audience/critic rating. **(Target Variable)**
* `Votes`: The total number of users who voted for the movie.
* `Director`: The main director of the film.
* `Actor 1`, `Actor 2`, `Actor 3`: The lead cast members.
## 🛠️ Project Workflow
1. **Data Cleaning:** * Removing duplicates and handling missing values in columns like `Rating`, `Duration`, and `Age`.
   * Parsing data types (e.g., converting text-based duration like "140 min" into a clean integer numeric format).
2. **Exploratory Data Analysis (EDA):** * Analyzing trends over time (e.g., how average ratings change across decades).
   * Identifying top-rated genres, most successful directors, and highly voted movies using visual charts.
3. **Feature Engineering & Transformation:** * Managing high-cardinality categorical data like `Director` and `Actors` (Target Encoding or Frequency Encoding).
   * Utilizing One-Hot Encoding for the `Genre` categories.
4. **Model Training:** Training regression models to accurately predict continuous ratings:
   * Linear Regression
   * Decision Tree Regressor
   * Random Forest Regressor
5. **Model Evaluation:** Measuring model performance using regression metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.
## 💻 Tech Stack Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
## 📈 Key Insights
* **Votes Correlation:** Movies with a higher number of user votes generally exhibit more reliable and stable ratings.
* **Genre Trends:** Specific genres or genre combinations consistently score higher averages compared to others.
* **Best Model Performance:** *[Tip: Fill in your final model evaluation details here, e.g., "The Random Forest Regressor achieved an R² score of 0.76 with an RMSE of 0.45."]*
## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
