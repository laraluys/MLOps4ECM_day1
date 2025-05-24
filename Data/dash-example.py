import pandas as pd
import numpy as np
import plotly.express as px
import dash

from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
from datetime import datetime

app = dash.Dash(__name__)
dataframe = pd.read_csv("datasets/water_potability/water_potability.csv", delimiter=";") 
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(children=[
    
    html.H1("Dashboard water potability", style={'text-align': 'center', "font-family":"Ariel"}),

    html.Label("select metric: ", htmlFor="slct-type", style={"font-family":"Ariel"}),
    dcc.Dropdown(
        id="slct_type",
        options=[
            {"label": "ph", "value": "ph"},
            {"label": "Solids", "value": "Solids"},
            {"label": "Sulfate", "value": "Sulfate"},
            {"label": "Conductivity", "value": "Conductivity"},
            {"label": "Organic_carbon", "value": "Organic_carbon"},
            {"label": "Trihalomethanes", "value": "Trihalomethanes"},
            {"label": "Turbidity", "value": "Turbidity"},
            {"label": "Hardness_2", "value": "Hardness_2"},
            {"label": "Hardness_1", "value": "Hardness_1"},
            {"label": "Potability", "value": "Potability"}],
        multi=False,
        value="ph",
        style={'width': "30%", 'display':'inline-block'}),

    dcc.Graph(id='data', figure={}),
])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='data', component_property='figure')],
    [Input(component_id='slct_type', component_property='value')]
)
def update_graph(option_type_slctd):

    dff = dataframe.copy()
    indexes = range(0,len(dff))
    # Plotly Express
    fig_all = px.line(x=indexes, y=dff[option_type_slctd], labels={'x':'index', 'y': option_type_slctd}, title=option_type_slctd)


    return fig_all,

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run()