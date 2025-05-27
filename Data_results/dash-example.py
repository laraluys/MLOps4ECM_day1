import pandas as pd
import plotly.express as px
import dash

from dash import dcc
from dash import html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
dataframe_train = pd.read_csv("Data_results/datasets/DehliClimate/DailyDelhiClimateTrain.csv", delimiter=",") 
dataframe_train["date"] = pd.to_datetime(dataframe_train['date'], format='%Y-%m-%d')

dataframe_train_copy = dataframe_train.copy()
dataframe_train_copy["month"] = dataframe_train["date"].dt.strftime("%m-%Y")
unique_months = dataframe_train_copy.drop_duplicates(subset='month', keep='first')

dataframe_test = pd.read_csv("Data_results/datasets/DehliClimate/DailyDelhiClimateTest.csv", delimiter=",") 
dataframe_test["date"] = pd.to_datetime(dataframe_test['date'], format='%Y-%m-%d')
correlation = dataframe_train.corr("spearman")
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(children=[
    
    html.H1("Dashboard of Dehli climate dataset", style={'text-align': 'center', "font-family":"Ariel"}),
    html.Div(id="dropdown", children=[

        html.Label("select time: ", htmlFor="slct-year",style={"font-family":"Ariel"}),
        dcc.RangeSlider(
            id="slct_year",
            min = 0,
            max = len(dataframe_train["date"]),
            value=[0, len(dataframe_train['date'])],
            marks = {index:row["month"] for index, row in unique_months.iterrows()}
        ),

        html.Label("select metric: ", htmlFor="slct-type", style={"font-family":"Ariel"}),
        dcc.Dropdown(
            id="slct_type",
            options=[
                {"label": "Mean temperature", "value": "meantemp"},
                {"label": "Humidity", "value": "humidity"},
                {"label": "Wind speed", "value": "wind_speed"},
                {"label": "Mean pressure", "value": "meanpressure"}],
            multi=False,
            value="meantemp",
            style={'width': "30%", 'display':'inline-block'}),
    ]),
    html.Div(id="graphs_data", children=[
        dcc.Graph(id='data', figure={}, style={'display':'inline-block'}),
        dcc.Graph(id='all_data', figure={}, style={'display':'inline-block'})
    ], style={"text-align":"center"}),

    html.Div(id="graphs_metrics", children=[
        dcc.Graph(id="correlation", figure={}, style={'display':'inline-block'}),
        dcc.Graph(id="boxplot", figure={}, style={'display':'inline-block'})
    ], style={"text-align":"center"})
]),

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='data', component_property='figure'),
     Output(component_id='all_data', component_property='figure'),
     Output(component_id='correlation', component_property='figure'),
     Output(component_id='boxplot', component_property='figure')],
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_type', component_property='value'),]
)
def update_graph(slctd_date, option_type_slctd):

    dff = dataframe_train.copy()
    dff_year = dff.loc[slctd_date[0]:slctd_date[1]]
    dff_type = pd.concat([dataframe_train.copy(), dataframe_test.copy()])

    # Plotly Express
    df_melt = dff_year.melt(id_vars="date", value_vars=["meantemp","humidity","wind_speed","meanpressure"])
    fig = px.line(df_melt, x="date", y="value", labels={'x':'date', 'y':'weather_values'}, color="variable", title="All climate data for the chosen year")
    fig_all = px.line(x=dff_type["date"], y=dff_type[option_type_slctd], labels={'x':'date', 'y': option_type_slctd}, title="All data for the chosen weather metric")

    fig_corr = px.imshow(correlation, text_auto=True, title="correlation matrix between weather metrics")
    fig_box = px.box(dff_type[option_type_slctd], title="box plot for the chosen weather metric")

    return fig , fig_all, fig_corr, fig_box

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run()