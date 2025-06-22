import os
from dashboard.lib.data import Dataset
from dashboard.views.FunctionsView import FunctionsView
from dashboard.views.ModelView import ModelView
from dashboard.views.ExamplesView import ExamplesView
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


#############################################################################

data = Dataset()

# Create the Dash app
assets_path = os.path.join(os.getcwd(),"assets")
app = Dash(
    __name__, 
    assets_folder=assets_path, 
    external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME]
)
app.title = "Embedding Visualization"

functions_view = FunctionsView(data)
model_view = ModelView(data)
examples_view = ExamplesView(data)

#############################################################################

app.layout = [
    dbc.Navbar(
        dbc.Container(
            html.A("Visual Analytics Tool for Sentence Embeddings Visualization v.0.4", className="logo")
        ),
    ),
    html.Br(),
    dbc.Container([
        dcc.Store(id="dcc-store-model-type"),
        dcc.Store(id="dcc-store-metric"),
        dcc.Store(id="dcc-store-composition"),
        dcc.Store(id="dcc-store-similarity"),
        dcc.Store(id="dcc-store-aggregation"),
        dcc.Store(id="dcc-store-model"),
        dcc.Store(id="dcc-store-layer"),
        dcc.Store(id="dcc-store-gap-bin"),
        dcc.Store(id="dcc-store-gap-bin-filter"),
        dcc.Store(id="dcc-store-idx"),
        dbc.Row([
            dbc.Col(
                [
                    dbc.Card(functions_view.getHeatmap(),
                        body=True,
                        className="mb-12",
                    ),
                    html.Br(),
                    dbc.Card(functions_view.getBoxPlots(),
                        body=True,
                        className="mb-12",
                        id=functions_view.getFunctionsDistributionsComponentId(),
                        style={"display": "none"},                        
                    ),
                ]
                , md = 2
            ),
            dbc.Col(
                [      
                    dbc.Card(model_view.getFunctionsInfo(),
                        body=True,
                        className="mb-12",
                        id=model_view.getFunctionsInfoComponentId(),
                        style={"display": "none"},
                    ),
                    html.Br(),
                    dbc.Card(model_view.getBarplot(),
                        body=True,
                        className="mb-12",
                        id=model_view.getBarPlotComponentId(),
                        style={"display": "none"},
                    ),
                    html.Br(),
                    dbc.Card(model_view.getLayersBarplot(),
                        body=True,
                        className="mb-12",
                        id=model_view.getLayerBarPlotComponentId(),
                        style={"display": "none"},
                    ),
                ], md = 4
            ),            
            dbc.Col(
                [
                    dbc.Card(examples_view.getModelInfo(),
                        body=True,
                        className="mb-12",
                        id=examples_view.getModelBadgeComponentId(),
                        style={"display": "none"},
                    ),
                    html.Br(),
                    dbc.Card(examples_view.getLayerErrorGapInfo(),
                        body=True,
                        className="mb-12",
                        id=examples_view.getLayerBadgeComponentId(),
                        style={"display": "none"},
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card(
                                examples_view.getLayersHeatmap(),  
                                body=True,
                                className="mb-12",
                                id=examples_view.getErrorGapComponentId(),
                                style={"display": "none"},
                            ), md=6
                        ),
                        dbc.Col(
                            dbc.Card(
                                examples_view.getLayersBarplot(),
                                body=True,
                                className="mb-12",
                                id=examples_view.getErrorGapDetailComponentId(),
                                style={"display": "none"},
                            ), md=6
                        ),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card(
                                examples_view.getTable(),  
                                body=True,
                                className="mb-12",
                                id=examples_view.getExamplesTableComponentId(),
                                style={"display": "none"},
                            ), md=9
                        ),
                        dbc.Col(
                            dbc.Card(
                                examples_view.getIdxCrossData(),
                                body=True,
                                className="mb-12",
                                id=examples_view.getExamplesTableHeatMapComponentId(),
                                style={"display": "none"},
                            ), md=3
                        ),                        
                    ]),                               
                ], md = 6
            ),
        ]),       
    ])
]

#############################################################################


@app.callback(
    Output("dcc-store-model-type", "data"),
    Output("dcc-store-metric", "data"),
    Input(functions_view.getDropDownModelTypeId(), "value"),
    Input(functions_view.getDropDownCorrelationMetricsId(), "value"),
)
def update_functions_heatmap_dropdowns(model_type, metric):
    return model_type, metric

# @app.callback(
#     Output("dcc-store-composition", "data"),
#     Output("dcc-store-similarity", "data"),    
#     Input(functions_view.getDropDownCompositionId(), "value"),
#     Input(functions_view.getDropDownSimilarityId(), "value"),
# )
# def update_functions_boxplots_dropdowns(composition, similarity):
#     return composition, similarity

@app.callback(
    Output("dcc-store-aggregation", "data"),
    Input(model_view.getDropDownAggregationId(), "value"),
)
def update_aggregation_dropdowns(aggregation):
    return aggregation

# @app.callback(
#     Output("dcc-store-model", "data"),
#     Input(model_view.getDropDownAggregationId(), "value"),
# )
# def update_models_dropdowns(aggregation, model):
#     return aggregation, model

#############################################################################


@app.callback(
    Output("dcc-store-composition", "data"),
    Output("dcc-store-similarity", "data"),
    Input(functions_view.getHeatmapId(), "clickData"),
)
def update_functions_dropdowns_by_heatmap_selection(click_data):
    if click_data:
        point = click_data["points"][0]
        functions_view.set_composition(point["x"])
        functions_view.set_similarity(point["y"])

        return functions_view.get_composition(), functions_view.get_similarity()
    else:
        return None, None

@app.callback(
    Output("dcc-store-layer", "data"),
    Output("dcc-store-gap-bin", "data"),
    Input(examples_view.getLayersHeatmapId(), "clickData"),
    Input(model_view.getBarplotId(), "clickData"),
)
def update_examples_layer_and_gap_by_heatmap_selection(heatmap_click_data, bar_click_data):
    ctx = callback_context

    if not ctx.triggered:
        return no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == examples_view.getLayersHeatmapId() and heatmap_click_data:
        point = heatmap_click_data["points"][0]
        examples_view.set_layer(point["x"])
        examples_view.set_error_gap(point["y"])
        return examples_view.get_layer(), examples_view.get_error_gap()

    elif triggered_id == model_view.getBarplotId() and bar_click_data:
        point = bar_click_data["points"][0]
        # model_view.set_model(point["y"])
        return None, None

    return no_update


@app.callback(
    Output("dcc-store-model", "data"),
    Input(model_view.getBarplotId(), "clickData"),
    Input(functions_view.getHeatmapId(), "clickData"),
)
def update_model_selection(bar_click_data, heatmap_click_data):
    ctx = callback_context

    if not ctx.triggered:
        return no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == model_view.getBarplotId() and bar_click_data:
        point = bar_click_data["points"][0]
        model_view.set_model(point["y"])
        return model_view.get_model()

    elif triggered_id == functions_view.getHeatmapId() and heatmap_click_data:
        point = heatmap_click_data["points"][0]
        return None

    return no_update


@app.callback(
    Output("dcc-store-idx", "data"),
    Output(examples_view.getTableId(), "active_cell"),
    Output(examples_view.getTableId(), "selected_cells"),
    Input(examples_view.getTableId(), "active_cell"),
    State(examples_view.getTableId(), "data"),
    Input(examples_view.getLayersHeatmapId(), "clickData"),
    Input(model_view.getBarplotId(), "clickData"),
)
def update_examples_dropdowns_by_heatmap_selection(active_cell, table_data, heatmap_click_data, bar_click_data):
    ctx = callback_context

    if not ctx.triggered:
        return no_update, no_update, no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == examples_view.getTableId() and active_cell and table_data:
        row_index = active_cell["row"]
        selected_row = table_data[row_index]
        idx = selected_row.get("idx")
        examples_view.set_idx(idx)

        return examples_view.get_idx(), no_update, no_update

    elif triggered_id in [examples_view.getLayersHeatmapId(), model_view.getBarplotId()]:
        if active_cell is not None:
            # Clear the active cell and selected cells
            return None, None, []

    return no_update, no_update, no_update

#############################################################################


@app.callback(
    Output(functions_view.getHeatmapId(), "figure"),
    Input("dcc-store-model-type", "data"),
    Input("dcc-store-metric", "data"),
)
def update_functions_heatmap(model_type, metric):
    return functions_view.createHeatmap()

@app.callback(
    Output(functions_view.getCompositionBoxplotId(), "figure"),
    Output(functions_view.getSimilarityBoxplotId(), "figure"),
    Output(functions_view.getCompositionSimilarityBoxplotId(), "figure"),
    Input("dcc-store-model-type", "data"),
    Input("dcc-store-metric", "data"),
    Input("dcc-store-composition", "data"),
    Input("dcc-store-similarity", "data"),
)
def update_functions_boxplots(model_type, metric, composition, similarity):
    functions_view.set_model_type(model_type)
    functions_view.set_metric(metric)
    functions_view.set_composition(composition)
    functions_view.set_similarity(similarity)

    return functions_view.createCompositionBoxplot(), functions_view.createSimilarityBoxplot(), functions_view.createCompositionSimilarityBoxplot()

@app.callback(
    Output(model_view.getCompositionBadgeId(), "children"),        
    Input("dcc-store-composition", "data"),
)
def update_composition_function_badge(comp):
    return comp

@app.callback(
    Output(model_view.getSimilarityBadgeId(), "children"),        
    Input("dcc-store-similarity", "data"),
)
def update_similarity_function_badge(sim):
    return sim

@app.callback(
    Output(model_view.getBarplotId(), "figure"),
    Output(model_view.getLayersBarplotId(), "figure"),
    Input("dcc-store-metric", "data"),
    Input("dcc-store-composition", "data"),
    Input("dcc-store-similarity", "data"),
    Input("dcc-store-aggregation", "data"),
    Input("dcc-store-model", "data"),
    Input("dcc-store-model-type", "data"),
)
def update_models_modelsbarplot(metric, composition, similarity, aggregation, model, model_type):
    model_view.set_metric( metric )
    model_view.set_composition( composition )
    model_view.set_similarity( similarity )
    model_view.set_aggregation( aggregation )
    model_view.set_model( model )
    model_view.set_model_type( model_type )

    return model_view.createBarplot(), model_view.createLayersBarplot()


@app.callback(
    Output(examples_view.getLayersHeatmapId(), "figure"),
    Input("dcc-store-composition", "data"),
    Input("dcc-store-similarity", "data"),
    Input("dcc-store-model", "data"),
)
def update_examples_heatmap(composition, similarity, model):
    examples_view.set_composition( composition )
    examples_view.set_similarity( similarity )
    examples_view.set_model( model )
    return examples_view.createLayersHeatmap()


@app.callback(
    Output(examples_view.getModelBadgeId(), "children"),        
    Input("dcc-store-model", "data"),
)
def update_examples_badges_model_info(model):
    examples_view.set_model( model )

    return examples_view.get_model()

@app.callback(
    Output(examples_view.getLayerBadgeId(), "children"),
    Output(examples_view.getGapBadgeId(), "children"),
    Input("dcc-store-model", "data"),
    Input("dcc-store-layer", "data"),
    Input("dcc-store-gap-bin", "data"),
)
def update_examples_badges(model, layer, gap):
    examples_view.set_model( model )
    examples_view.set_layer( layer )
    if gap is not None:
        examples_view.set_error_gap( gap )

    return examples_view.get_layer(), examples_view.get_error_gap_range()


@app.callback(
    Output(examples_view.getLayersBarplotId(), "figure"),
    Input("dcc-store-layer", "data"),
    Input("dcc-store-gap-bin", "data"),
)
def update_examples_barplot(layer, gap):
    examples_view.set_layer( layer )
    examples_view.set_error_gap( gap )
    return examples_view.createLayerBarplot()


@app.callback(
    Output(examples_view.getTableId(), "data"),
    Input("dcc-store-gap-bin", "data"),
)
def update_examples_table(gap):
    examples_view.set_error_gap( gap )
    return examples_view.createTableData().to_dict(orient="records")


@app.callback(
    Output(examples_view.getCrossDataHeatmapId(), "figure"),
    Input("dcc-store-idx", "data"),
)
def update_examples_heatmap2(idx):
    examples_view.set_idx( idx )
    return examples_view.createCrossDataHeatmap()


#############################################################################


@app.callback(
    Output(functions_view.getFunctionsDistributionsComponentId(), "style"),
    Input("dcc-store-composition", "data"),
    Input("dcc-store-similarity", "data"),
)
def toggle_visibility_distributions_component(comp, sim):
    if comp is not None and sim is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}
    

@app.callback(
    Output(model_view.getBarPlotComponentId(), "style"),
    Input("dcc-store-composition", "data"),
    Input("dcc-store-similarity", "data"),
)
def toggle_visibility_barplot_component(comp, sim):
    if comp is not None and sim is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}

@app.callback(
    Output(model_view.getFunctionsInfoComponentId(), "style"),
    Input("dcc-store-composition", "data"),
    Input("dcc-store-similarity", "data"),
)
def toggle_visibility_model_badge_component(comp, sim):
    if comp is not None and sim is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}

@app.callback(
    Output(model_view.getLayerBarPlotComponentId(), "style"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_layer_barplot_component(model):
    if model is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}



@app.callback(
    Output(examples_view.getModelBadgeComponentId(), "style"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_model_badge_component(model):
    if model is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output(examples_view.getErrorGapComponentId(), "style"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_error_gap_component(model):
    if model is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output(examples_view.getLayerBadgeComponentId(), "style"),
    Input("dcc-store-layer", "data"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_layer_badge_component(layer, model):
    if model is None:
        return {"display": "none"}    
    elif layer is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output(examples_view.getErrorGapDetailComponentId(), "style"),
    Input("dcc-store-layer", "data"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_error_gap_detail_component(layer, model):
    if model is None:
        return {"display": "none"}    
    elif layer is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output(examples_view.getExamplesTableComponentId(), "style"),
    Input("dcc-store-gap-bin", "data"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_examples_table_component(layer, model):
    if model is None:
        return {"display": "none"}
    elif layer is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output(examples_view.getExamplesTableHeatMapComponentId(), "style"),
    Input("dcc-store-idx", "data"),
    Input("dcc-store-model", "data"),
)
def toggle_visibility_examples_table_component(layer, model):
    if model is None:
        return {"display": "none"}
    elif layer is not None:
        return {"display": "block"}
    else:
        return {"display": "none"}

#############################################################################


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
