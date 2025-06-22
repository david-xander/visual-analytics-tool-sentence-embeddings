import os
from dashboard.lib.data import Dataset
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc


class ModelView():
    def __init__(self, data:Dataset):
        self.data = data
        self.metric = "spearman"
        self.composition = None
        self.similarity = None
        self.aggregation = "max"
        self.model = None
        self.model_type = "both"
    
    def getDropDownModelOptions(self):
        df = self.data.get_correlation_dataframe()
        df = self.filter_by_functions_and_model_type(df)
        return df["model"].unique()

    def filter_by_functions_and_model_type(self, df):
        df = df[(df["composition"]==self.get_composition()) & (df["similarity"]==self.get_similarity())]
        if self.get_model_type() != "both":
            if self.model_type == "static":
                df = df[df["type"]=="Static"]
            elif self.model_type == "transformer":
                df = df[df["type"]=="Transformer"]            
        return df

    def createBarplot(self):
        if self.get_composition() is None or self.get_similarity() is None:
            return None
        
        df = self.data.get_correlation_dataframe()
        df = self.filter_by_functions_and_model_type(df)

        if self.get_aggregation() == "max":
            df = df.groupby(["model", "compsim", "type"]).apply(
                lambda x: pd.Series({
                    self.get_metric(): x[self.get_metric()].max(),
                })
            ).reset_index()
        else:
            df = df.groupby(["model", "compsim", "type"]).apply(
                lambda x: pd.Series({
                    self.get_metric(): x[self.get_metric()].mean(),
                })
            ).reset_index()

        df = df.sort_values(by="model", ascending=False)

        fig = px.histogram(
            df,
            x=self.get_metric(),
            y="model",
            color = "type",
            title="",
            color_discrete_sequence=["#003f5c", "#bc5090"],
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            height=450,
            showlegend=False,
        )
        fig.update_traces(
            hovertemplate=self.get_metric().capitalize()+": %{x}<extra></extra>",
        )

        return fig
    
    def createLayersBarplot(self):
        if self.get_model() is None:
            return None

        df = self.data.get_correlation_dataframe()
        df = self.filter_by_functions_and_model_type(df)
        df =  df[df["model"]==self.get_model()]

        # Extract data for plotting
        x_values = df["layer"]
        y_values = df[self.get_metric()]
        

        # Create bar plot using go
        fig = go.Figure(
            data=go.Bar(
                x=x_values,
                y=y_values,
                marker=dict(color="#003f5c"),
            )
        )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            height=250,  # Adjust height if needed
            xaxis=dict(
                title="Layers of "+self.get_model(),
                tickmode='linear',  # Ensure all layers are shown as ticks
                tick0=0,
                dtick=1  # Step by 1 for each layer
            ),
            yaxis=dict(
                title=self.get_metric().capitalize(),  # Use metric name as y-axis title
            ),
        )

        return fig
    
    def getBarplot(self):
        firstColumn = [
            dbc.Row([
                dbc.Col([],md=6),
                dbc.Col([
                    dcc.Dropdown(
                        id=self.getDropDownAggregationId(),
                        options=[
                            {"label": "Aggregated by Maximum", "value": "max"},
                            {"label": "Aggregated by Mean", "value": "mean"},
                        ],
                        clearable=False,
                        value=self.get_aggregation(),
                    ),
                ],
                md=6),               
            ]),
            dcc.Graph(id=self.getBarplotId(), figure=self.createBarplot()),
        ]
        return firstColumn

    def getLayersBarplot(self):
        secondColumn = [
            # dbc.Row([
            #     dbc.Col([
            #         dcc.Dropdown(
            #             id=self.getDropDownModelId(),
            #             options=self.getDropDownModelOptions(),
            #             clearable=False,
            #             value=self.get_model(),
            #         ),
            #     ], md=12),      
            # ]),
            # html.Br(),
            dcc.Graph(id=self.getLayersBarplotId(), figure=self.createLayersBarplot()),             
        ]
        return secondColumn


    def getFunctionsInfo(self):
        column = [
            dbc.Row([
                dbc.Col([
                    html.H4([
                        "Composition: ",
                        dbc.Badge(id=self.getCompositionBadgeId(), className="bg-info"),                   
                    ]),                    
                ], md=6, className="text-center"),
                dbc.Col([
                    html.H4([
                        "Similarity: ",
                        dbc.Badge(id=self.getSimilarityBadgeId(), className="bg-info"),                   
                    ]),                    
                ], md=6, className="text-center"),                
            ]),
        ]
        return column     


    def getBarplotId(self):
        return "modelsBarplot"

    def getLayersBarplotId(self):
        return "modelLayersBarplot"
    
    def getDropDownAggregationId(self):
        return "dropDownAggregation"

    def getDropDownModelId(self):
        return "dropDownModel"

    def getCompositionBadgeId(self):
        return "modelCompositionInfo"

    def getSimilarityBadgeId(self):
        return "modelSimilarityInfo"

    def getBarPlotComponentId(self):
        return "componentModelBarPlot"
    
    def getLayerBarPlotComponentId(self):
        return "componentModelLayerBarPlot"


    def getFunctionsInfoComponentId(self):
        return "componentFunctionsInfo"

    def set_metric(self, value):
        self.metric = value

    def get_metric(self):
        return self.metric

    def set_composition(self, value):
        self.composition = value
    
    def get_composition(self):
        return self.composition
    
    def set_similarity(self, value):
        self.similarity = value    
    
    def get_similarity(self):
        return self.similarity
    
    def set_aggregation(self, value):
        self.aggregation = value    
    
    def get_aggregation(self):
        return self.aggregation    
    
    def set_model(self, value):
        self.model = value    
    
    def get_model(self):
        return self.model        

    def set_model_type(self, value):
        self.model_type = value    
    
    def get_model_type(self):
        return self.model_type