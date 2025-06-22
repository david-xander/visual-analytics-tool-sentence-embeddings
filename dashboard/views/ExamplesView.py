import os
from dashboard.lib.data import Dataset
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import numpy as np

class ExamplesView():
    def __init__(self, data:Dataset):
        self.data = data
        self.composition = None
        self.similarity = None
        self.model = None
        self.model_type = "Transformer"
        self.layer = None
        self.idx = None
        self.dimensionality_reduction = "pca"
        self.error_gap = None
        self.error_gap_filer = None
    
#############################################################################

    def getDropDownLayerOptions(self):
        df = self.data.get_correlation_dataframe()
        df = self.filter_by_model(df)
        return [{"label": f"Layer {layer}", "value": layer} for layer in df["layer"].unique()]
    
    def getExampleData(self):
        df = self.createTableData()
        df = df[df["idx"]==self.get_idx()]
        return df

    def get_num_layers(self):
        return len(self.getDropDownLayerOptions())
    
    def get_composition_similarity_layer(self):
        res = self.get_similarity()+self.get_composition()+str(self.get_layer())
        if self.get_model_type() != "Transformer":
            res = self.get_similarity()+self.get_composition()
        return res

    def get_composition_layer_for_similarity(self, similarity):
        res = similarity+self.get_composition()+str(self.get_layer())
        if self.get_model_type() == "Static":
            res = similarity+self.get_composition()
        return res

    def filter_by_model(self, df):
        df = df[(df["model"]==self.get_model())]
        return df
    
    def filter_by_gap_error(self, df):
        low = float(self.get_error_gap())
        up = float(self.get_error_gap())+0.1

        if up >= 1:
            df = df[(df["gap"]>=low) & (df["gap"]<=1)]
        else:
            df = df[(df["gap"]>=low) & (df["gap"]<up)]        

        return df        
    
    def extract_components_raw_string(self, string):
        if self.get_model_type() == "Static":
            pattern = r"([A-Z][a-z]+)([A-Za-z_]+)$"
            match = re.match(pattern, string)
            if match:
                similarity_name = match.group(1)# "Cos", "Euc", "Icm"
                composition_function = match.group(2) # "F_inf", "Sum"
                layer = 1
                return composition_function, similarity_name, layer
            else:
                return "error"
        else:
            pattern = r"([A-Z][a-z]+)([A-Za-z_]+)(\d+)$"
            match = re.match(pattern, string)
            if match:
                similarity_name = match.group(1)# "Cos", "Euc", "Icm"
                composition_function = match.group(2) # "F_inf", "Sum"
                layer = int(match.group(3))

                return composition_function, similarity_name, layer
            else:
                return "error"
        
    def reduce_with_pca(self, vectors, n_components=3):
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(vectors)
        return reduced_vectors

    def reduce_with_tsne(self, vectors, n_components=3, perplexity=10, random_state=42):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        reduced_vectors = tsne.fit_transform(vectors)
        return reduced_vectors

#############################################################################

    def createLayersHeatmap(self):
        if self.get_model() is None:
            return None

        df = self.data.get_metrics_dataframe(model=self.get_model())
        data = []

        # Define bin edges and labels
        bin_edges = [i * 0.1 for i in range(10)] + [1.1]  # [0, 0.1, ..., 0.9, 1.1]
        bin_labels = [round(bin_edges[i], 1) for i in range(len(bin_edges) - 1)]  # [0.0, 0.1, ..., 0.9]
        bin_labels[-1] = 0.9  # Ensure the last label is 0.9 for the range [0.9, 1.0]

        for layer in range(0, self.get_num_layers()):
            colx = f"{self.get_similarity()}{self.get_composition()}{layer}"
            if self.get_model_type() == "Static":
                colx = f"{self.get_similarity()}{self.get_composition()}"

            temp = abs(df["label_ok"] - df[colx])
            temp_binned = pd.cut(temp, bins=bin_edges, labels=bin_labels, include_lowest=True)
            temp_df = pd.DataFrame({
                "gap": temp_binned,
                "layer": layer
            })
            data.append(temp_df)

        combined_df = pd.concat(data)

        custom_colorscale = [
            [0.0, "white"],
            [1.0, "#003f5c"],
        ]

        fig = px.density_heatmap(
            combined_df, 
            x="layer", 
            y="gap",
            nbinsx=self.get_num_layers(),
            color_continuous_scale=custom_colorscale,
        )

        fig.update_layout(
            xaxis_title="Layers of "+self.get_model(),
            yaxis_title="Absolute Error Gap",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(
                tickmode="linear",
                tick0=0,
                dtick=1,
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=bin_labels,
                ticktext=[str(label) for label in bin_labels],
            ),
            height=300,
        )

        return fig

    def createLayerBarplot(self):
        if self.get_error_gap() is None:
            return None
        
        df = self.data.get_metrics_dataframe(model=self.get_model())
        df["gap"] = abs(df["label_ok"] - df[f"{self.get_composition_similarity_layer()}"])
        
        df = self.filter_by_gap_error(df)

        fig = px.histogram(
            df,
            y="gap",
            color_discrete_sequence=["#003f5c"],         
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(
                title="Absolute Error Gap",
                nticks=10,
                # tickmode="linear",                
                # tick0=df["gap"].min(),
                # dtick=0.01,
            ),
            xaxis=dict(
                title="Ocurrences",
            ),            
            bargap=0.2,
            height=300,
        )

        return fig    

    def createTableData(self):
        if self.get_error_gap() is None:
            empty_dict = []
            ecols = ["idx", "sentence1", "sentence2", "label", "gapcos", "gapeuc","gapicm"]
            empty_df = pd.DataFrame(empty_dict, columns=ecols)
            return empty_df
        
        df = self.data.get_metrics_dataframe(model=self.get_model())
        df["gap"] = abs(df["label_ok"]-df[self.get_composition_similarity_layer()]) 
        
        cols = ["idx","sentence1","sentence2","label_ok","gap"]
        if self.get_composition() != "Cos":
            cols.append("gapCos")
            df["gapCos"] = df["label_ok"]-df[self.get_composition_layer_for_similarity("Cos")]
        if self.get_composition() != "Euc":
            cols.append("gapEuc")
            df["gapEuc"] = df["label_ok"]-df[self.get_composition_layer_for_similarity("Euc")]
        if self.get_composition() != "Icm":
            cols.append("gapIcm")
            df["gapIcm"] = df["label_ok"]-df[self.get_composition_layer_for_similarity("Icm")]

        df = df[cols].rename(columns={"label_ok": "label"}).round(2)

        df = self.filter_by_gap_error(df)
        df = df.sort_values(by="gap", ascending=False)        
        
        return df
    
    def createCrossDataHeatmap(self):
        if self.get_idx() is None:
            return None

        df = self.data.get_metrics_dataframe(model=self.get_model())
        df = df[df["idx"]==self.get_idx()]

        cols = set(df.columns) - set(["sentence1","sentence2","label","idx","label_ok","model"])

        data = []
        for col in cols:
            composition_function, similarity_name, layer = self.extract_components_raw_string(col)
            actual = df[col].iloc[0]
            label = df["label_ok"].iloc[0]
            data.append(
                    {
                        "model": df["model"],
                        "sentence1": df["sentence1"],
                        "sentence2": df["sentence2"],
                        "label": label,
                        "composition": composition_function,
                        "similarity": similarity_name,
                        "layer": layer,
                        "gap": float(label) - float(actual)
                    }
            )
        

        df = pd.DataFrame(data)

        if self.get_model_type() == "Static":
            df = df[df["composition"]!="Cls"]
        else:
            df = df[df["layer"]==self.get_layer()]
        
        df = df.pivot_table(index="composition",columns="similarity", values="gap")

        custom_colorscale = [
            [0.0, "#e74c3c "],
            [0.5, "#9FE2BF"],
            [1.0, "#e74c3c"],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale=custom_colorscale,
                zmid=0, 
                hoverongaps=False,
                # hoverinfo="none",
                colorbar=dict(
                    title="Error Gap",
                    # titleside="right",
                    # ticks="outside",
                ),
            )
        )
        fig.update_layout(
            xaxis_title="Similarity Function",
            yaxis_title="Composition Function",
            margin=dict(l=40, r=40, t=40, b=40),
            height=250,
        )
        return fig

    def createScatterPlot(self):
        df1 = self.data.get_metrics_dataframe(model=self.get_model())[["idx","sentence1","sentence2","label_ok", f"{self.get_composition_similarity_layer()}"]].rename(columns={"label_ok":"label", f"{self.get_composition_similarity_layer()}": "predicted"})
        df1["gap"] = df1["label"]-df1["predicted"]

        df1 = self.filter_by_gap_error(df1)

        df2 = self.data.get_embeddings_dataframe(model=self.get_model())
        df2 = df2[(df2["compf"]==self.get_composition().lower()) & (df2["layer"]==self.get_layer())]
        df2 = df2.rename(columns={"sentence1":"embeddings1", "sentence2":"embeddings2"})

        df = pd.merge(df1, df2, on='idx', how='left')

        # Combine sentence1 and sentence2 vectors into a single array
        sentence1_vectors = torch.stack([t.detach() for t in df["embeddings1"]]).numpy()
        sentence2_vectors = torch.stack([t.detach() for t in df["embeddings2"]]).numpy()

        vectors = np.concatenate([sentence1_vectors, sentence2_vectors], axis=0)

        if len(vectors.shape) == 3:
            vectors = vectors.reshape(vectors.shape[0], -1)

        df_s1 = df[["idx","sentence1","label","predicted","gap"]].rename(columns={"sentence1":"sentence"}).reset_index()
        df_s2 = df[["idx","sentence2","label","predicted","gap"]].rename(columns={"sentence2":"sentence"}).reset_index()
        df_s1["pair"] = "sentence1"
        df_s2["pair"] = "sentence2"
        df = pd.concat([df_s1, df_s2])

        reduced_vectors = []
        if self.get_dimensionality_reduction() == "pca":
            reduced_vectors = self.reduce_with_pca(vectors)
        else:
            reduced_vectors = self.reduce_with_tsne(vectors)
        

        fig = px.scatter_3d(
            df,
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            color="pair",
            hover_data=["idx","sentence","label","predicted","gap"],
        )
        fig.update_layout(
            height=800,
        )

        unique_idx = df["idx"].unique()
        for idx in unique_idx:
            # Find the two points with this idx
            first_point_idx = df[df["idx"] == idx].index[0]  # First occurrence
            second_point_idx = first_point_idx + len(df) // 2  # Corresponding in second half

            # Fetch coordinates for the two points
            x = [reduced_vectors[first_point_idx, 0], reduced_vectors[second_point_idx, 0]]
            y = [reduced_vectors[first_point_idx, 1], reduced_vectors[second_point_idx, 1]]
            z = [reduced_vectors[first_point_idx, 2], reduced_vectors[second_point_idx, 2]]

            gap = df.loc[first_point_idx, "gap"].iloc[0]
            predicted = df.loc[first_point_idx, "predicted"].iloc[0]
            line_width = 1 + abs(gap) * 10  # Scales to [1, 5]

            # Add a line trace connecting the points
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='lines',
                    line=dict(color='black', width=line_width),
                    name = "",
                    hovertext=f"Idx: {idx}, Gap: {gap:.2f}, Predicted: {predicted:.2f}",
                    showlegend=False,
                )
            )

        return fig

#############################################################################

    def getModelInfo(self):
        column = [
            dbc.Row([
                dbc.Col([
                    html.H4([
                        dbc.Badge(id=self.getModelBadgeId(), className="bg-info"),                   
                    ]),                    
                ], md=12, className="text-center"),
            ]),
        ]
        return column     

    def getLayerErrorGapInfo(self):
        column = [
            dbc.Row([
                dbc.Col([
                    html.H4([
                        "Layer: ", 
                        dbc.Badge(id=self.getLayerBadgeId(), 
                                  className="bg-info"),
                        " - Error gap: ",
                        dbc.Badge(id=self.getGapBadgeId(), 
                                  className="bg-info"),                        
                    ]),
                ],md=12, className="text-center"),     
            ]),
        ]
        return column

    def getLayersHeatmap(self):
        colum = [
            dcc.Graph(id=self.getLayersHeatmapId(), figure=self.createLayersHeatmap()),
        ]
        return colum

    def getLayersBarplot(self):
        colum = [
            dcc.Graph(id=self.getLayersBarplotId(), figure=self.createLayerBarplot()),
        ]
        return colum
    
    def getTable(self):
        colum = [
            dash_table.DataTable(
                id = self.getTableId(),
                data=self.createTableData().to_dict(orient="records"),
                columns=[
                    {"name": col, "id": col} for col in ["idx","sentence1", "sentence2", "label", "gapCos","gapEuc","gapIcm"]
                ],
                page_size=12, 
                style_table={'overflowX': 'auto'},
                style_data={
                    'fontSize': '12px',
                    'fontFamily': 'sans-serif',
                    'whiteSpace': 'normal',
                    'height': 'auto',             
                },
                style_header={
                    'fontSize': '12px',
                    'fontFamily': 'sans-serif',
                    'fontWeight': 'bold',
                },
                style_data_conditional=[
                    {'if': {'column_id': 'idx'}, 'textAlign':'left'},
                    {'if': {'column_id': 'sentence1'}, 'textAlign': 'left'},
                    {'if': {'column_id': 'sentence2'}, 'textAlign': 'left'},
                    {'if': {'column_id': 'label'}, 'textAlign': 'right'},
                    {'if': {'column_id': 'gap'}, 'textAlign': 'right'},

                    # Conditional formatting for gapCos
                    {
                        'if': {'filter_query': '{gapCos} >= -0.1 && {gapCos} <= 0.1', 'column_id': 'gapCos'},
                        'backgroundColor': '#9FE2BF',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '({gapCos} > 0.1 && {gapCos} <= 0.3) || ({gapCos} >= -0.3 && {gapCos} < -0.1)', 'column_id': 'gapCos'},
                        'backgroundColor': '#FFBF00',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '({gapCos} > 0.3 && {gapCos} <= 0.6) || ({gapCos} >= -0.6 && {gapCos} < -0.3)', 'column_id': 'gapCos'},
                        'backgroundColor': '#FF7F50',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{gapCos} > 0.6 || {gapCos} < -0.6', 'column_id': 'gapCos'},
                        'backgroundColor': '#DE3163',
                        'color': 'white',
                    },

                    # Repeat for gapEuc
                    {
                        'if': {'filter_query': '{gapEuc} >= -0.1 && {gapEuc} <= 0.1', 'column_id': 'gapEuc'},
                        'backgroundColor': '#9FE2BF',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '({gapEuc} > 0.1 && {gapEuc} <= 0.3) || ({gapEuc} >= -0.3 && {gapEuc} < -0.1)', 'column_id': 'gapEuc'},
                        'backgroundColor': '#FFBF00',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '({gapEuc} > 0.3 && {gapEuc} <= 0.6) || ({gapEuc} >= -0.6 && {gapEuc} < -0.3)', 'column_id': 'gapEuc'},
                        'backgroundColor': '#FF7F50',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{gapEuc} > 0.6 || {gapEuc} < -0.6', 'column_id': 'gapEuc'},
                        'backgroundColor': '#DE3163',
                        'color': 'white',
                    },

                    # Repeat for gapIcm
                    {
                        'if': {'filter_query': '{gapIcm} >= -0.1 && {gapIcm} <= 0.1', 'column_id': 'gapIcm'},
                        'backgroundColor': '#9FE2BF',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '({gapIcm} > 0.1 && {gapIcm} <= 0.3) || ({gapIcm} >= -0.3 && {gapIcm} < -0.1)', 'column_id': 'gapIcm'},
                        'backgroundColor': '#FFBF00',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '({gapIcm} > 0.3 && {gapIcm} <= 0.6) || ({gapIcm} >= -0.6 && {gapIcm} < -0.3)', 'column_id': 'gapIcm'},
                        'backgroundColor': '#FF7F50',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{gapIcm} > 0.6 || {gapIcm} < -0.6', 'column_id': 'gapIcm'},
                        'backgroundColor': '#DE3163',
                        'color': 'white',
                    },                    

                ],
            ),
        ]
        return colum    

    def getIdxCrossData(self):
        colum = [        
            dcc.Graph(id=self.getCrossDataHeatmapId(), figure=self.createCrossDataHeatmap()),
        ]
        return colum

    def getScatterPlot(self):
        colum = [
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id=self.getDropDownDimensionalityReductionId(),
                        options=[
                            {"label": "PCA", "value": "pca"},
                            {"label": "T-SNE", "value": "tsne"},
                            {"label": "UMAP", "value": "umap"},
                        ],
                        clearable=False,
                        value=self.get_dimensionality_reduction(),
                    ),                    
                ],md=6),
                dbc.Col([],md=6),
            ]),
            dcc.Graph(id=self.getScatterPlotId(), figure=self.createScatterPlot()),
        ]
        return colum


#############################################################################

    def getLayersHeatmapId(self):
        return "exampleLayersHeatmap"
    
    def getLayersBarplotId(self):
        return "layersBarplot"
    
    def getModelBadgeId(self):
        return "modelBadge"

    def getLayerBadgeId(self):
        return "layerBadge"

    def getGapBadgeId(self):
        return "gapBadge"

    def getTableId(self):
        return "examplesDataTable"   

    def getCrossDataHeatmapId(self):
        return "crossDataHeatmap"    
    
    def getScatterPlotId(self):
        return "scatterPlot"

    def getDropDownDimensionalityReductionId(self):
        return "dropDownDimensionalityReduction"
    
    def getModelBadgeComponentId(self):
        return "componentExamplesModelBadge"

    def getLayerBadgeComponentId(self):
        return "componentExamplesLayerBadge"

    def getGapBadgeComponentId(self):
        return "componentExamplesGapBadge"

    def getErrorGapComponentId(self):
        return "componentExamplesErrorGap"
    
    def getErrorGapDetailComponentId(self):
        return "componentExamplesErrorGapDetail"

    def getExamplesTableComponentId(self):
        return "componentExamplesTable"

    def getExamplesTableHeatMapComponentId(self):
        return "componentExamplesTableHeatMap"

#############################################################################


    def set_composition(self, value):
        self.composition = value
    
    def get_composition(self):
        return self.composition
    
    def set_similarity(self, value):
        self.similarity = value    
    
    def get_similarity(self):
        return self.similarity 
    
    def set_model(self, value):
        if value is not None:
            self.model = value   
            df = self.data.get_correlation_dataframe()
            df = df[df["model"]==value]
            self.set_model_type(df["type"].iloc[0])
    
    def get_model(self):
        return self.model  

    def set_model_type(self, value):
        self.model_type = value    
    
    def get_model_type(self):
        return self.model_type           
    
    def set_layer(self, value):
        self.layer = value    
    
    def get_layer(self):
        return self.layer
    
    def set_idx(self, value):
        self.idx = value

    def get_idx(self):
        return self.idx
    
    def set_dimensionality_reduction(self, value):
        self.dimensionality_reduction = value

    def get_dimensionality_reduction(self):
        return self.dimensionality_reduction
    
    def set_error_gap(self, value):
        self.error_gap = value

    def get_error_gap(self):
        return self.error_gap    
    
    def get_error_gap_range(self):
        if self.error_gap is None:
            return None
        
        e = float(self.error_gap)
        e_up = e + 0.1
        r = f"{e:.1f} < {e_up:.1f}"
        return r

    def set_error_gap_filter(self, value):
        self.error_gap_filter = value

    def get_error_gap_filter(self):
        return self.error_gap_filer