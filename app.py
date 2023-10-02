import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index2.html')

# Define a route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        # Map the numeric class to its corresponding label (e.g., 'setosa', 'versicolor', 'virginica')
        if prediction == 0:
            predicted_class = 'Setosa'
        elif prediction == 1:
            predicted_class = 'Versicolor'
        else:
            predicted_class = 'Virginica'

        return render_template('index2.html', prediction=predicted_class)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)

