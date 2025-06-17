import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import numpy as np
import dash_bootstrap_components as dbc
import pandas as pd

# Load model and scaler
model = joblib.load('models/house_price_model.pkl')
feature_names = joblib.load('models/model_features.pkl')
#scaler = joblib.load('scaler.pkl')

# Dash app setup
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    html.H2("🏠 House Price Predictor", className="mt-4 mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Overall Quality"),
            dcc.Input(id='input-quality', type='number', value=5, className='form-control')
        ], width=6),
        dbc.Col([
            html.Label("GrLivArea (sq ft)"),
            dcc.Input(id='input-grliv', type='number', value=1500, className='form-control')
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Garage Cars"),
            dcc.Input(id='input-garage', type='number', value=2, className='form-control')
        ], width=6),
        dbc.Col([
            html.Label("Total Basement SF"),
            dcc.Input(id='input-bsmt', type='number', value=800, className='form-control')
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(html.Button('Predict', id='submit-val', n_clicks=0, className='btn btn-primary btn-block'), width=12)
    ], className="mb-4"),

    html.Div(id='prediction-output', style={'fontSize': 22, 'fontWeight': 'bold'})
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-quality', 'value'),
    State('input-grliv', 'value'),
    State('input-garage', 'value'),
    State('input-bsmt', 'value')
)
def predict_price(n_clicks, overall_qual, grliv, garage_cars, bsmt):
    if n_clicks > 0:
        # Step 1: Capture user inputs into a dictionary
        user_input = {
            'OverallQual': overall_qual,
            'GrLivArea': grliv,
            'GarageCars': garage_cars,
            'TotalBsmtSF': bsmt
        }

        # Step 2: Create a full feature vector with 0s for missing ones
        full_input = {feat: user_input.get(feat, 0) for feat in feature_names}

        # Step 3: Convert to DataFrame with the correct column order
        input_df = pd.DataFrame([full_input])

        # Step 4: Predict
        prediction = model.predict(input_df)[0]

        return f"🏷️ Predicted House Price: ${prediction:,.2f}"

    return ''

if __name__ == '__main__':
    app.run(debug=True)
