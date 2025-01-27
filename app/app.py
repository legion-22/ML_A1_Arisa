import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Mappings
name_mapping = {
    'Ambassador': 0, 'Ashok': 1, 'Audi': 2, 'BMW': 3, 'Chevrolet': 4,
    'Daewoo': 5, 'Datsun': 6, 'Fiat': 7, 'Force': 8, 'Ford': 9,
    'Honda': 10, 'Hyundai': 11, 'Isuzu': 12, 'Jaguar': 13, 'Jeep': 14,
    'Kia': 15, 'Land': 16, 'Lexus': 17, 'MG': 18, 'Mahindra': 19,
    'Maruti': 20, 'Mercedes-Benz': 21, 'Mitsubishi': 22, 'Nissan': 23,
    'Opel': 24, 'Peugeot': 25, 'Renault': 26, 'Skoda': 27, 'Tata': 28,
    'Toyota': 29, 'Volkswagen': 30, 'Volvo': 31
}

fuel_mapping = {'Diesel': 1, 'Petrol': 2}
seller_type_mapping = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
transmission_mapping = {'Manual': 1, 'Automatic': 2}
owner_mapping = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Car Price Prediction for Chaky's Company"),
    html.P("Instruction"),
    html.P("In order to predict car price, you need to choose car brand, year, kilometers driven, fuel type, seller type, transmission, owner type, mileage, engine, maximum power and number of seat"),
    html.P("If you don't know some data, you can empty this data."),

    # Input form
    html.Div([
        html.Label("Car Brand:", style={"marginBottom": "10px"}),
        dcc.Dropdown(
            id="name",
            options=[{"label": name, "value": encoded} for name, encoded in name_mapping.items()],
            placeholder="Select Car Name",
            style={"marginBottom": "20px"}  # Add margin below the dropdown
        ),
        html.Br(),

        html.Label("Year:", style={"marginBottom": "10px"}),
        dcc.Input(id="year", type="number", placeholder="e.g., 2015", style={"marginBottom": "20px"}),
        html.Br(),

        html.Label("Kilometers Driven:", style={"marginBottom": "10px"}),
        dcc.Input(id="km_driven", type="number", placeholder="e.g., 50000", style={"marginBottom": "20px"}),
        html.Br(),

        html.Label("Fuel Type:", style={"marginBottom": "10px"}),
        dcc.Dropdown(
            id="fuel",
            options=[{"label": k, "value": v} for k, v in fuel_mapping.items()],
            placeholder="Select Fuel Type",
            style={"marginBottom": "20px"}
        ),
        html.Br(),

        html.Label("Seller Type:", style={"marginBottom": "10px"}),
        dcc.Dropdown(
            id="seller_type",
            options=[{"label": k, "value": v} for k, v in seller_type_mapping.items()],
            placeholder="Select Seller Type",
            style={"marginBottom": "20px"}
        ),
        html.Br(),

        html.Label("Transmission:", style={"marginBottom": "10px"}),
        dcc.Dropdown(
            id="transmission",
            options=[{"label": k, "value": v} for k, v in transmission_mapping.items()],
            placeholder="Select Transmission Type",
            style={"marginBottom": "20px"}
        ),
        html.Br(),

        html.Label("Owner Type:", style={"marginBottom": "10px"}),
        dcc.Dropdown(
            id="owner",
            options=[{"label": k, "value": v} for k, v in owner_mapping.items()],
            placeholder="Select Owner Type",
            style={"marginBottom": "20px"}
        ),
        html.Br(),

        html.Label("Mileage (kmpl):", style={"marginBottom": "10px"}),
        dcc.Input(id="mileage", type="number", placeholder="e.g., 20", style={"marginBottom": "20px"}),
        html.Br(),

        html.Label("Engine (CC):", style={"marginBottom": "10px"}),
        dcc.Input(id="engine", type="number", placeholder="e.g., 1197", style={"marginBottom": "20px"}),
        html.Br(),

        html.Label("Max Power (bhp):", style={"marginBottom": "10px"}),
        dcc.Input(id="max_power", type="number", placeholder="e.g., 82", style={"marginBottom": "20px"}),
        html.Br(),

        html.Label("Seats:", style={"marginBottom": "10px"}),
        dcc.Input(id="seats", type="number", placeholder="e.g., 5", style={"marginBottom": "20px"}),
        html.Br(),

        html.Button("Predict", id="predict-button", n_clicks=0)
    ], style={"width": "50%", "margin": "0 auto"}),

    # Output
    html.Div(id="output-prediction", style={"marginTop": "20px", "textAlign": "center"})
])

# Callback for prediction
@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("name", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
)

def predict_price(n_clicks, name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats):
    if n_clicks > 0:
        # Handle missing values
        inputs = [name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]
        inputs = [0 if v is None else v for v in inputs]

        # Only scale the numerical features: km_driven, mileage, engine, max_power
        numerical_features = inputs[2:6]  # km_driven, mileage, engine, max_power
        scaled_numerical_features = scaler.transform([numerical_features])  # Scale only these 4

        # Replace the numerical features with their scaled versions
        inputs[2:6] = scaled_numerical_features[0]

        # Predict
        log_price = model.predict([inputs])[0]
        price = np.exp(log_price)  # Apply inverse log transform
        return f"The predicted price of the car is: {price:.2f} Bath"
    
    return "Please fill in the details and click Predict."

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
