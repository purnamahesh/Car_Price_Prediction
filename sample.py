from flask import Flask, request, render_template, jsonify, url_for, redirect
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

algo = ['Gradient Boosting', 'Extreme Gradient Boosting',
        'Random Forest', 'Lasso', 'Ridge', 'Linear']

file_names = [
    'Gradient_Boosting_Regressor', 'Extreme_Gradient_Boosting_Regressor', 'Random_Forest_Regressor',
    'Lasso_Regression', 'Ridge_Regression', 'Linear_Regression'
]
models = []
for file_name in file_names:
    models.append(pickle.load(open(f"models/{file_name}.pkl", "rb")))

row = None
fuel_values = ['CNG' 'Diesel' 'LPG' 'Petrol']
seller_type_values = ['Dealer' 'Individual' 'Trustmark Dealer']
transmission_values = ['Automatic' 'Manual']
owner_values = ['First Owner' 'Fourth & Above Owner' 'Second Owner' 'Test Drive Car'
                'Third Owner']
brand_values = ['Ambassador' 'Ashok' 'Audi' 'BMW' 'Chevrolet' 'Daewoo' 'Datsun' 'Fiat'
                'Force' 'Ford' 'Honda' 'Hyundai' 'Isuzu' 'Jaguar' 'Jeep' 'Kia' 'Land'
                'Lexus' 'MG' 'Mahindra' 'Maruti' 'Mercedes-Benz' 'Mitsubishi' 'Nissan'
                'Opel' 'Renault' 'Skoda' 'Tata' 'Toyota' 'Volkswagen' 'Volvo']


@app.route('/')
def home():
    return '<h2>Hello World</h2>'


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    print(request_data)
    year = request_data['year']
    km_driven = request_data['km_driven']
    fuel = request_data['fuel']
    seller_type = request_data['seller_type']
    transmission = request_data['transmission']
    owner = request_data['owner']
    mileage = float(request_data['mileage'])
    engine = request_data['engine']
    max_power = request_data['max_power']
    brand = request_data['brand']
    row = [
        year, km_driven, fuel, seller_type, transmission,
        owner, mileage, engine, max_power, brand
    ]

    df = pd.DataFrame([pd.Series(row)])
    predictions = [model.predict(df) for model in models]
    print(predictions)

    return jsonify({
        "X": {
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "brand": brand
        },
        "Prediction": prediction
    })


@app.route('/predict', methods=['GET'])
def get_row():
    return jsonify({"row": row})


app.run(debug=True)
