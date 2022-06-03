
from flask import Flask, request, render_template



import joblib
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

model = joblib.load(open('model2.pkl', 'rb'))
cv = joblib.load(open('cv1.pkl', 'rb'))

#ddevjyoti karan huhh this is time taking



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['message']
        data = [text]
        vectorizer = cv.transform(data).toarray()
        prediction = model.predict(vectorizer)
        return render_template('result.html',prediction=prediction)
    
    '''if prediction:
        return render_template('index.html', prediction_text='The review is Postive')
    else:
        return render_template('index.html', prediction_text='The review is Negative.')'''
        




if __name__ == "__main__":
    app.run(debug=True)