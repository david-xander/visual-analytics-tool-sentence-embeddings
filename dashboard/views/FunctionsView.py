import os
from dashboard.lib.data import Dataset
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


class FunctionsView():
    def __init__(self, data:Dataset):
        self.data = data
        self.metric = "spearman"
        self.model_type = "both"
        self.composition = None
        self.similarity = None
    
    def filter_by_model_type(self, df):
        if self.model_type == "static":
            df = df[df["type"]=="Static"]
            df = df[df["composition"]!="Cls"]
        elif self.model_type == "transformer":
            df = df[df["type"]=="Transformer"]
        else:
            df = df[df["composition"]!="Cls"]
        return df        

    def createHeatmap(self):
        df = self.data.get_correlation_dataframe()
        df = self.filter_by_model_type(df)
        df = df.pivot_table(index="similarity", columns="composition", values=self.get_metric())

        custom_colorscale = [
            [0.0, "#003f5c"],   # most negative values
            [0.5, "white"],  # midpoint (close to zero)
            [1.0, "#003f5c"],   # most positive values
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale=custom_colorscale,
                zmid=0,  # Ensures 0 is the neutral midpoint for diverging scale        
                hoverongaps=False,
                colorbar=dict(
                    title=self.get_metric().capitalize(),
                    # titleside="right",
                    # ticks="outside",
                ),
            )
        )
        fig.update_layout(
            # title=f"Performance considering {self.get_metric()} correlation",
            xaxis_title="Composition Function",
            yaxis_title="Similarity Function",
            margin=dict(l=40, r=40, t=40, b=40),
            height=250,
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    def createCompositionBoxplot(self):
        if self.get_composition() is None or self.get_similarity() is None:
            return None
                
        df = self.data.get_correlation_dataframe()     
        df = self.filter_by_model_type(df)
        df = df[df["composition"] == self.get_composition()]

        fig = go.Figure()
        for similarity in df['similarity'].unique():
            filtered_data = df[df['similarity'] == similarity]
            fig.add_trace(
                go.Box(
                    y=filtered_data[self.get_metric()],
                    name=similarity,
                    boxpoints="all",  # Show all points
                    jitter=0.3,  # Spread the points for better visibility
                    pointpos=-1.8,  # Position points relative to the box
                )
            )
        
        # Update layout
        fig.update_layout(
            # title=f"{self.get_composition()}. performance across different similarity functions",
            xaxis_title="Similarity Function",
            yaxis_title=f"{self.get_metric()}",
            xaxis=dict(tickangle=0),
            margin=dict(l=40, r=40, t=40, b=40),
            height=200,
        )
        
        return fig  

    def createSimilarityBoxplot(self):
        if self.get_composition() is None or self.get_similarity() is None:
            return None
                
        df = self.data.get_correlation_dataframe()   
        df = self.filter_by_model_type(df)
        df = df[df["similarity"] == self.get_similarity()]

        fig = go.Figure()
        for composition in df['composition'].unique():
            filtered_data = df[df['composition'] == composition]
            fig.add_trace(
                go.Box(
                    y=filtered_data[self.get_metric()],
                    name=composition,
                    boxpoints="all",  # Show all points
                    jitter=0.3,  # Spread the points for better visibility
                    pointpos=-1.8,  # Position points relative to the box
                )
            )
        
        # Update layout
        fig.update_layout(
            # title=f"{self.get_similarity()}. performance across different composition functions",
            xaxis_title="Composition Function",
            yaxis_title=f"{self.get_metric()}",
            xaxis=dict(tickangle=45),
            margin=dict(l=40, r=40, t=40, b=40),
            height=250,
        )
        
        return fig
    
    def createCompositionSimilarityBoxplot(self):
        if self.get_composition() is None or self.get_similarity() is None:
            return None
        
        df = self.data.get_correlation_dataframe()  
        df = self.filter_by_model_type(df)
        df = df[(df["composition"] == self.get_composition()) & (df["similarity"] == self.get_similarity())]

        fig = go.Figure()
        fig.add_trace(
            go.Box(
                x=df[self.get_metric()],
                y=df["compsim"],
                orientation="h",  # Horizontal orientation
                boxpoints="all",  # Show all points
                jitter=0.3,  # Spread the points for better visibility
                pointpos=0,  # Position points at 0
                name="Filtered Data"  # Trace name
            )
        )
        
        # Update layout
        fig.update_layout(
            # title=f"{self.get_composition()} + {self.get_similarity()} performance",
            xaxis_title=f"{self.get_metric()}",
            yaxis_title=f"{self.get_similarity()} + {self.get_composition()}",
            xaxis=dict(tickangle=0),  # Rotate x-axis labels
            margin=dict(l=40, r=40, t=40, b=40),
            height=150,  # Adjust plot height
        )
        
        return fig    
    
    def getHeatmap(self):
        firstColumn = [
            dbc.Row([
                dbc.Col([
                    # dbc.Label("Model type"),
                    dcc.Dropdown(
                        id=self.getDropDownModelTypeId(),
                        options=[
                            {"label": "static + transformers", "value": "both"},
                            {"label": "static", "value": "static"},
                            {"label": "transformer", "value": "transformer"},
                        ],
                        clearable=False,
                        value=self.get_model_type(),
                    ),
                ], 
                md=8),
                dbc.Col([
                    # dbc.Label("Correlation"),
                    dcc.Dropdown(
                        id=self.getDropDownCorrelationMetricsId(),
                        options=[
                            {"label": "Spearman", "value": "spearman"},
                            {"label": "Pearson", "value": "pearsoncor"},
                        ],
                        clearable=False,
                        value=self.get_metric(),
                    ),
                ],
                md=4),               
            ]),
            # html.Br(),
            dcc.Graph(id=self.getHeatmapId(), figure=self.createHeatmap()),
        ]
        return firstColumn

    def getBoxPlots(self):
        secondColumn = [
            # dbc.Row([
            #     dbc.Col([
            #         # dbc.Label("Similarity"),
            #         dcc.Dropdown(
            #             id=self.getDropDownSimilarityId(),
            #             options=[
            #                 {"label": "Cosine (Cos)", "value": "Cos"},
            #                 {"label": "Euclidean (Euc)", "value": "Euc"},
            #                 {"label": "ICM (Icm)", "value": "Icm"},
            #             ],
            #             clearable=False,
            #             value=self.get_similarity(),
            #         ),
            #     ], md=6),                
            #     dbc.Col([
            #         # dbc.Label("Composition"),
            #         dcc.Dropdown(
            #             id=self.getDropDownCompositionId(),
            #             options=[
            #                 {"label": "Avg", "value": "Avg"},
            #                 {"label": "Cls", "value": "Cls"},
            #                 {"label": "F_ind", "value": "F_ind"},
            #                 {"label": "F_inf", "value": "F_inf"},
            #                 {"label": "F_joint", "value": "F_joint"},
            #                 {"label": "Sum", "value": "Sum"},
            #             ],
            #             clearable=False,
            #             value=self.get_composition(),
            #         ),
            #     ], md=6),            
            # ]),
            # html.Br(),
            dcc.Graph(id=self.getCompositionBoxplotId(), figure=self.createCompositionBoxplot()),
            dcc.Graph(id=self.getSimilarityBoxplotId(), figure=self.createSimilarityBoxplot()), 
            dcc.Graph(id=self.getCompositionSimilarityBoxplotId(), figure=self.createCompositionSimilarityBoxplot()),                       
        ]
        return secondColumn
    
    def getHeatmapId(self):
        return "mainHeatmap"

    def getCompositionBoxplotId(self):
        return "compositionBoxplot"

    def getSimilarityBoxplotId(self):
        return "similarityBoxplot"

    def getCompositionSimilarityBoxplotId(self):
        return "compositionSimilarityBoxplot"        

    def getDropDownModelTypeId(self):
        return "dropDownModelType"

    def getDropDownCorrelationMetricsId(self):
        return "dropDownCorrelationMetrics"    

    def getDropDownCompositionId(self):
        return "dropDownComposition"

    def getDropDownSimilarityId(self):
        return "dropDownSimilarity"

    def getFunctionsDistributionsComponentId(self):
        return "componentFunctionsDistributions"
    
    def set_metric(self, value):
        self.metric = value

    def set_model_type(self, value):
        self.model_type = value
    
    def get_metric(self):
        return self.metric
    
    def get_model_type(self):
        return self.model_type

    def set_composition(self, value):
        self.composition = value
    
    def set_similarity(self, value):
        self.similarity = value
    
    def get_composition(self):
        return self.composition
    
    def get_similarity(self):
        return self.similarity