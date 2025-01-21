# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

import plotly.graph_objects as go
import plotly.express as px

from config import med_color_map
from src.reconstruction import get_XYZ_of_surface
from src.util import timing
from src.data.interpolation import SmoothedInterpolator

import pandas as pd
import numpy as np

import config


def get_line_chart_for_metric(df: pd.DataFrame,
                              ivom_timeline: pd.DataFrame,
                              metric,
                              prognosis_df: pd.DataFrame = None,
                              selected_drug=None,
                              current_date=None,
                              main_date=None,
                              compare_date=None,
                              conversion_factor=1.0,
                              y_title=None, ):
    """Returns a line chart for a metric. The chart shows the measured values, the smoothed values, the forecasting and
    the IVOMs. The chart is a plotly figure.
    :param df: The dataframe containing the metric.
    :param ivom_timeline: The dataframe containing the IVOMs.
    :param metric: The metric to plot. Can be one of the columns of the dataframe.
    :param prognosis_df: The dataframe containing the forecasting. If None, no forecasting is shown.
    :param selected_drug: The drug for which the forecasting is shown. If None, all drugs are shown.
    :param current_date: The current date. If None, no arrow is shown.
    :param main_date: The main date. If None, no arrow is shown.
    :param compare_date: The compare date. If None, no arrow is shown.
    :param conversion_factor: The conversion factor for the metric. Default is 1.0. For metrics like volume, the
        conversion factor can be 1_000_000 to accomodate for very small values.
    :param y_title: The title of the y-axis. If None, the metric is used.
    :return: A plotly figure."""
    if config.DatabaseKeys.drug not in ivom_timeline.columns:
        ivom_timeline[config.DatabaseKeys.drug] = "Control"
    maximum_value = max(df[metric].dropna()) / conversion_factor
    minimum_value = min(df[metric].dropna()) / conversion_factor
    fig = go.Figure()
    arrows = []
    for drug in ivom_timeline[config.DatabaseKeys.drug].unique():
        temp_df = ivom_timeline[ivom_timeline[config.DatabaseKeys.drug] == drug]
        for i, row in temp_df.iterrows():
            fig.add_shape(
                type="rect",
                x0=row[config.DatabaseKeys.visit_date],
                y0=minimum_value - 0.5,
                x1=row[config.DatabaseKeys.visit_date] + pd.DateOffset(2),
                y1=maximum_value + 0.5,
                line=dict(
                    color="rgba(0,0,0,0)",
                ),
                fillcolor=med_color_map[drug],
                opacity=0.35,
                layer="below"
            )
    temp_df = df.dropna(subset=[metric])
    fig.add_trace(go.Scatter(
        x=temp_df.index,
        y=temp_df[metric] / conversion_factor,
        mode='markers',
        showlegend=False,
        hovertemplate="<i>Measured value</i>: %{y:.2f}",
        visible=True,
        opacity=0.5
    ))
    date_range = pd.date_range(temp_df.index[-1], temp_df.index[0], freq='1d')
    new_df = pd.DataFrame(index=date_range)
    new_df.loc[temp_df.index, metric] = temp_df[metric]
    new_df = new_df.interpolate(method='time')
    fig.add_trace(go.Scatter(
        x=date_range,
        y=new_df[metric] / conversion_factor,
        mode='lines',
        hoverinfo='skip',
        showlegend=False,
        line=dict(color='rgba(31, 119, 180, 0.3)'),
    )
    )
    for date in [current_date, main_date, compare_date]:
        if date is None:
            continue
        try:
            y_pos = new_df.loc[date, metric] / conversion_factor
        except KeyError:
            y_pos = minimum_value
        arrow = go.layout.Annotation(dict(
            x=date,
            y=y_pos,
            xref="x", yref="y",
            text="Current date" if date == current_date else "Main VOL" if date == main_date else "Compare VOL",
            showarrow=True,
            axref="x", ayref='y',
            ax=date,
            ay=maximum_value + 0.1,
            arrowhead=3,
            arrowwidth=1.5,
            arrowcolor='rgb(255,51,0)',
        )
        )
        arrows += [arrow]
    interpolator = SmoothedInterpolator(moving_average_window=90,
                                        fit_dates=temp_df.index,
                                        fit_values=temp_df[metric] / conversion_factor, )

    smoothed = interpolator(date_range)
    fig.add_trace(go.Scatter(
        x=date_range,
        y=smoothed,
        name=f'{metric}',
        mode='lines',
        hovertemplate="<i>Smoothed value</i>: %{y:.2f}",
        fill="tonexty",
        fillcolor='rgba(31, 119, 180, 0.2)',
        showlegend=False,
        line=dict(color='rgba(31, 119, 180, 1.0)'),
    ))
    for drug in ivom_timeline[config.DatabaseKeys.drug].unique():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=10, color=med_color_map[drug], symbol="square"),
                                 name=drug,
                                 showlegend=True, ))
    if prognosis_df is not None:
        temp_df = df.dropna(subset=[metric])
        prognosed_dates = [temp_df.index[0]] + [temp_df.index[0] + pd.DateOffset(months=i) for i in [1, 3, 6, 9, 12]]
        selected_drug = selected_drug if type(selected_drug) is list else [selected_drug]
        for drug in selected_drug:
            drug_data = prognosis_df.loc[drug]
            last_value = temp_df.iloc[0][metric] / conversion_factor
            values = [last_value] + [predicted / conversion_factor for predicted in drug_data.values[:5]]
            fig.add_trace(go.Scatter(
                x=prognosed_dates,
                y=values,
                name=drug,
                mode='lines+markers',
                line=dict(color=med_color_map[drug], dash="dash"),
                visible=True,
            ))
    # Only show the last 12 months
    fig.update_xaxes(range=[df[config.DatabaseKeys.visit_date].iloc[0] - pd.DateOffset(months=12),
                            df[config.DatabaseKeys.visit_date].iloc[0] + pd.DateOffset(months=1)])
    fig.update_yaxes(range=[minimum_value - 0.1, maximum_value + 0.3])  # Add some space for the arrows

    fig.update_layout(
        annotations=arrows,
        xaxis=dict(title=''),
        yaxis=dict(title=y_title if y_title is not None else metric),
        height=300,
        legend=dict(
            yanchor="bottom",
            y=0.0,
            xanchor="left",
            x=1.0
        ),
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        hovermode="x",

    )
    return fig


def get_ivom_timeline(df, oct_dates):
    """Returns a timeline of the IVOMs. The timeline is a plotly figure."""
    drug_key = config.DatabaseKeys.drug
    if config.DatabaseKeys.drug not in df.columns:
        vis_date_frame = pd.DataFrame(index=df.index)
        vis_date_frame[drug_key] = "Control"
    else:
        vis_date_frame = df[[drug_key]].copy()
        vis_date_frame.loc[vis_date_frame[drug_key].isna(), drug_key] = "Control"
    for date in oct_dates:
        vis_date_frame.loc[date, drug_key] = "Today"
    vis_date_frame["visdate_end"] = pd.to_datetime(vis_date_frame.index) + pd.DateOffset(1)
    vis_date_frame["ones"] = np.ones(len(vis_date_frame)) * 3
    color_map = med_color_map.copy()
    color_map["Today"] = "white"
    fig = px.timeline(vis_date_frame,
                      x_start=vis_date_frame.index,
                      x_end="visdate_end",
                      y="ones",
                      color=config.DatabaseKeys.drug,
                      title="Uploads",
                      color_discrete_map=med_color_map,
                      hover_data={
                          "visdate_end": False,
                          drug_key: False,
                          config.DatabaseKeys.drug: True
                      }
                      )
    fig.update_layout(
        xaxis=dict(title=''),
        yaxis=dict(title='', showticklabels=False),
        showlegend=False,
        width=800,
        height=75,
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=50
        )
    )
    return fig


def show_image(oct_slice, mask=None, segment=False, height=None):
    """Shows an image of an OCT slice with a mask overlay. If segment is True, the different layers are shown as
    well."""
    if mask is None:
        label_mask = np.zeros(oct_slice.shape)
    else:
        label_mask = np.vectorize(lambda x: config.LAYERS_TO_STRING[x])(mask)
    fig = px.imshow(oct_slice, origin='lower', binary_string=True, )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False, visible=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, visible=False, zeroline=False)
    fig.update_layout(
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        height=700 if height is None else height,
    )
    if mask is not None:
        fig.update(data=[{"customdata": label_mask,
                          'hovertemplate': "Prediction: %{customdata}"}], )
    if segment:
        for i, layer in config.LAYERS_TO_STRING.items():
            if i == 0:
                continue
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=f'rgb{config.LAYERS_TO_COLOR[i]}', symbol="square"),
                name=layer,
                showlegend=True,
            ))
        fig.update_layout(legend=dict(
            orientation="h",
            title="Legend",
            yanchor="top",
            y=0,
            xanchor="left",
            x=0),
        )

    return fig


def show_volume(vol, height=None):
    fig = px.imshow(vol.oct, animation_frame=0, origin='lower', binary_string=True)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False, visible=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, visible=False, zeroline=False)
    fig.update_layout(
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        height=700 if height is None else height,
    )
    return fig


@timing
def get_3D_plot(vol):
    """Returns a 3D plot of the reconstructed objects and the not reconstructed objects."""
    mins = {"x": float('inf'), "y": float('inf'), "z": float('inf')}
    maxs = {"x": 0, "y": 0, "z": 0}
    reconstructed_labels = vol.reconstructed_objects.keys()
    not_reconstructed_labels = [label for label in vol.labels if label not in reconstructed_labels and label != 0]
    fig = go.Figure()
    for label in not_reconstructed_labels:
        oh_mask = (vol.masks == label)
        X, Y, Z = get_XYZ_of_surface(vol.grid, oh_mask, downsample=20)
        mins["x"] = min(mins["x"], min(X))
        mins["y"] = min(mins["y"], min(Y))
        mins["z"] = min(mins["z"], min(Z))
        maxs["x"] = max(maxs["x"], max(X))
        maxs["y"] = max(maxs["y"], max(Y))
        maxs["z"] = max(maxs["z"], max(Z))
        fig.add_trace(go.Mesh3d(
            x=X,
            y=Y,
            z=Z,
            alphahull=-1,
            name=f"{config.LAYERS_TO_STRING[label]}",
            showlegend=True,
            opacity=0.1,  # needs to be small to see through all surfaces
            color=f'rgb{config.LAYERS_TO_COLOR[label]}',
        ))
    for label in reconstructed_labels:
        for i, object in enumerate(vol.reconstructed_objects[label]):
            x, y, z = object.x, object.y, object.z
            fig.add_trace(go.Mesh3d(
                x=x,
                y=y,
                z=z,
                name=f"{config.LAYERS_TO_STRING[label]}",
                text=f"{object.get_volume() / 1_000_000:.2f} ml",
                alphahull=0,
                opacity=0.7,
                color=f'rgb{config.LAYERS_TO_COLOR[label]}',
                legendgroup=f"{config.LAYERS_TO_STRING[label]}",
                showlegend=i == 0,
            ))
    fig.update_layout(showlegend=True,
                      legend=dict(
                          orientation="h",
                          title="Legend",
                          yanchor="top",
                          y=0,
                          xanchor="left",
                          x=0),
                      scene=dict(
                          xaxis=dict(title="Width", range=[mins["x"] - 1, maxs["x"] + 1]),
                          yaxis=dict(title="Slices", tickvals=np.array(vol.grid)[:, 1],
                                     ticktext=np.arange(1, len(vol.grid) + 1),
                                     range=[mins["y"], maxs["y"]], title_text="Slices"),
                          zaxis=dict(title="Height", nticks=0, range=[0, vol.oct.shape[2]]),
                          aspectmode='cube'),
                      margin=go.layout.Margin(
                          l=0,
                          r=0,
                          b=0,
                          t=0
                      ),
                      height=700, width=1000
                      )

    return fig


def add_image_to_fig3D(fig3d, image, grid_position):
    """Adds an image to a 3D plot. The image is placed at the grid position."""
    xx = np.linspace(0, image.shape[1], image.shape[1])
    zz = np.linspace(0, image.shape[0], image.shape[0])
    xx, zz = np.meshgrid(xx, zz)

    yy = grid_position * np.ones(xx.shape)
    if type(fig3d["data"][-1]) is not go.Surface:
        fig3d.add_trace(go.Surface(x=xx,
                                   y=yy,
                                   z=zz,
                                   surfacecolor=image,
                                   colorscale="gray",
                                   cmin=0,
                                   cmax=255,
                                   showscale=False,
                                   opacity=1.0
                                   ))
    else:
        fig3d["data"][-1]['y'] = yy
        fig3d["data"][-1]['surfacecolor'] = image
    return fig3d
