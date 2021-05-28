import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib


# Loading the dataset
df = pd.read_csv('scrapped_data_rmp.csv', engine='python')

for i in range(len(df)):
    if df['quality'][i]=="awful":
        df['quality'][i]=1
    elif df['quality'][i]=='poor':
        df['quality'][i]=2
    elif df['quality'][i]=='average':
        df['quality'][i]=3
    elif df['quality'][i]=='good':
        df['quality'][i]=4
    elif df['quality'][i]=='awesome':
        df['quality'][i]=5
            

corpus = []

# Looping till 1000 because the number of rows are 1000
for i in range(0, 2345):
    # Removing the special character from the reviews and replacing it with space character
    review = re.sub('[^a-zA-Z]', ' ', df['comment'][i])

    # Converting the review into lower case character
    review = review.lower()

    # Tokenizing the review by words
    review_words = review.split()

    # Removing the stop words using nltk stopwords
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

    # Stemming the words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]

    # Joining the stemmed words
    review = ' '.join(review)
    
    # Creating a corpus
    corpus.append(review)


# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 2].values
y1=y.astype(int)


# Creating a pickle file for the CountVectorizer model
joblib.dump(cv, "cv1.pkl")


# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)
pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
# Creating a pickle file for the Multinomial Naive Bayes model
joblib.dump(classifier, "model2.pkl")