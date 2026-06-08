import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# EXPLORE THE DATA
#1. firs we will upload the data 
df = pd.read_csv(r"C:\Users\user\Videos\Codsoft\Task 2 Movie Rating Prediction\IMDb Movies India.csv", encoding='latin1')
print(df.columns.tolist())
#2. once we have the data the first thing we do is to understand what we're working with. 
print (df.shape)
print (df.columns.tolist())
#3. then we will check for any missing values in the data.
print(df.info())
print(df.isnull().sum())
#4. now what does the target (Rating) look like?
#5. bins=20 means that we are dividing the range of Ratings into 20 equal parts and counting how many Ratings fall into each part (1 part= 0.45)
df['Rating'].hist(bins=20)
plt.title("Rating Distribution")
plt.show()
print(df.describe())


# CLEAN AND PREPROCESS THE DATA
#1. first we will drop the duplicate values if there are any.
df = df.drop_duplicates()
#2. Now fill or drop the missing values. We can either fill the missing values with the mean, median, or mode of the column, or we can drop the rows with missing values. Here we will fill the missing values with the mean of the column.
df['genre'] = df['genre'].fillna('Unknown')
df['overview'] = df[''].fillna('') # we will convert the overview column to a string and fill the missing values with an empty string.
df['budget'] = df['budget'].fillna(df['budget'].median())
#3. Now we will convert types of columns if necessary.
df['release_date'] = pd.to_datetime(df['release_date']).dt.year


# FEATURE ENGINEERING
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
#1. structured features: we will use the genre and release_date as features for our model.
structureed_features = ['budget', 'runtime', 'release_date', 'vote_count']
X_structured = df[structureed_features].fillna(0).values
#2. categorical features: we will use the genre as a categorical feature for our model.
df['genre_list'] = df['genre'].str.split('|')
mlb = MultiLabelBinarizer()
X_genre = mlb.fit_transform(df['genre_list'])
#3. text features: we will use the overview column as a text feature for our model.
tfidf = TfidfVectorizer(max_features=500, stop_words='english')# top 500 words frequent words 
X_test = tfidf.fit_transform(df['overview'])
#4. Now we will conbine everthing
X = sp.hstack([X_structured, X_genre, X_test])
y = df['Rating'].values


# NOW LETS BUILD THE MODEL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#1 basline model
baseline = Ridge()
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))


#  UPGRADING THE MODEL      
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF R²:", r2_score(y_test, y_pred_rf))


#  EVALUATE THE MODEL
#1. actual vs predicted
plt.scatter(y_test, y_pred_rf, alpha=0.4)
plt.plot([1,10], [1,10], 'r--') # perfect predictions line
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted")
plt.show()


#  PREDICTING ON A NEW MOVIE
def predict_Rating(budget, runtime, year, vote_count, genre, overview):
    structured = sp.csr_matrix([[budget, runtime, year, vote_count]])
    genre_encoded = mlb.transform([genre.split('|')])
    text_encoded = tfidf.transform([overview])
    features = sp.hstack([structured, genre_encoded, text_encoded])
    return rf.predict(features)[0]
#1 Example usage
predict_Rating(
    budget=50_000_000,
    runtime=120,
    year=2023,
    vote_count=5000,
    genre="Action|Adventure",
    overview="A hero must save the world from an alien invasion"
)