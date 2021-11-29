from flask import Flask, request, render_template, url_for
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)


gbr = pickle.load(open("models/Extreme_Gradient_Boosting_Regressor.pkl", "rb"))

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
    return render_template('index.html')


model_cols = ["year", "kmdriven", "fuel", "sellertype",
              "transmission", "owner", "mileage", "engine", "maxpower", "brand"]


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.form
    values = [request_data[col] for col in model_cols]
    print(values)
    """ final = [
        values[0], values[1],
        fuel_values[int(values[2])],
        seller_type_values[int(values[3])],
        transmission_values[int(values[4])],
        owner_values[int(values[5])],
        values[6],
        values[7],
        values[8],
        brand_values[int(values[9])]
    ] """
    df = pd.DataFrame([pd.Series(values)])
    prediction = "{:,.2f}".format(gbr.predict(df)[0])
    print(prediction)
    return render_template('index.html', y=prediction, cols=model_cols, x=values)


if __name__ == "__main__":
    app.run(debug=True)
