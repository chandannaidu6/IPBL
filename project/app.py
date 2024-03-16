from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    new_instance_2d = np.array(int_features).reshape(1, -1)
    prediction = model.predict(new_instance_2d)

    labels = ['Schimers1Lefteye', 'Schimers1Righteye', 'Schimers2Lefteye', 'Schimers2Righteye']

    result = zip(labels, prediction[0])

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run()
