from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
car = pd.read_csv("Cleaned-Car.csv")
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

@app.route('/')
def index():
    company = sorted(car['company'].unique())
    models = sorted(car['name'].unique())
    year = sorted(car['year'].unique())
    fuel_type = car['fuel_type'].unique()
    company.insert(0, "Select Company")
    
    return render_template('index.html', companies = company, car_models = models, years = year, fuel_types = fuel_type)

@app.route('/predict', methods =['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug = True)
    
    