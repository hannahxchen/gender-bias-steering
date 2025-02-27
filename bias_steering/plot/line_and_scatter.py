import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .plot_utils import COLORS


def plot_score_per_layer(
    proj_corr_MD, proj_corr_WMD, rmse_MD, rmse_WMD, model_names, 
    width=600, height=250, corr_range=[-0.1, 1], rmse_range=[0, 0.5],
):
    fig = make_subplots(
        rows=1, cols=4, horizontal_spacing=0.018,
        subplot_titles=model_names, shared_yaxes=True,
        specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]]
    )

    showlegend = True
    pos = [[1, 1], [1, 2], [1, 3], [1, 4]]
    for i, model in enumerate(model_names):
        layers = list(range(len(proj_corr_MD[i])))
        fig.add_trace(go.Scatter(
            x=layers, y=proj_corr_MD[i], mode="lines", name="MD", 
            marker_color=COLORS["MD"], showlegend=showlegend, legendgroup="r", legendgrouptitle_text="Correlation"
        ), secondary_y=False, row=pos[i][0], col=pos[i][1])
        fig.add_trace(go.Scatter(
            x=layers, y=proj_corr_WMD[i], mode="lines", name="WMD", 
            marker_color=COLORS["WMD"], showlegend=showlegend, legendgroup="r", legendgrouptitle_text="Correlation"
        ), secondary_y=False, row=pos[i][0], col=pos[i][1])

        fig.add_trace(go.Scatter(
            x=layers, y=rmse_MD[i], mode="lines", name="MD", line_dash="dot",
            marker_color=COLORS["MD"], showlegend=showlegend, legendgroup="rmse", legendgrouptitle_text="RMSE"
        ), secondary_y=True, row=pos[i][0], col=pos[i][1])
        fig.add_trace(go.Scatter(
            x=layers, y=rmse_WMD[i], mode="lines", name="WMD", line_dash="dot", 
            marker_color=COLORS["WMD"], showlegend=showlegend, legendgroup="rmse", legendgrouptitle_text="RMSE"
        ), secondary_y=True, row=pos[i][0], col=pos[i][1])

        showlegend = False

    fig.update_layout(
        plot_bgcolor='white', font=dict(size=14), 
        margin=dict(l=10, r=8, t=25, b=20), width=width, height=height,
    )

    fig.update_layout(legend=dict(
        yanchor="top", y=0.99, xanchor="right", x=1.08,
        grouptitlefont=dict(size=14), font=dict(size=14)
    ))

    fig.update_xaxes(
        mirror=True, showgrid=False,
        title_text="Layer", title_standoff=4, range=[0, None],
        zeroline = False, zerolinecolor='darkgrey',
        title_font=dict(size=16), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey',
    )
    fig.update_yaxes(
        title_standoff=5, mirror=True, showgrid=False, 
        zeroline = False, zerolinecolor='darkgrey',
        title_font=dict(size=16), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey',
    )
    fig.update_yaxes(title_text="Correlation (r)", range=corr_range, secondary_y=False, col=1)
    fig.update_yaxes(title_text="RMSE", range=rmse_range, title_standoff=8, secondary_y=True, col=4)
    fig.update_annotations(font=dict(size=15))
    fig.update_yaxes(showticklabels=False, secondary_y=True, col=1)
    fig.update_yaxes(showticklabels=False, secondary_y=True, col=2)
    fig.update_yaxes(showticklabels=False, secondary_y=True, col=3)
    fig.show()
    return fig


def plot_score_per_layer_single(
    proj_corr_MD, proj_corr_WMD, 
    y_title="Pearson Correlation (r)", title_text=None, 
    width=435, height=300, y_range=None, x_range=None, 
    showlegend=True, legend_x=1.0, legend_y=0.25
):
    fig = go.Figure()
    layers = list(range(len(proj_corr_MD)))
    fig.add_trace(go.Scatter(x=layers, y=proj_corr_MD, mode="markers+lines", name="MD", marker_color=COLORS["MD"], showlegend=showlegend))
    fig.add_trace(go.Scatter(x=layers, y=proj_corr_WMD, mode="markers+lines", name="WMD", marker_color=COLORS["WMD"], showlegend=showlegend))
    fig.update_layout(
        plot_bgcolor='white', font=dict(size=14), 
        title=dict(text=title_text, font=dict(size=18), x=0.55, y=0.98),
        margin=dict(l=10, r=10, t=25, b=30), width=width, height=height,
    )
    if showlegend:
        fig.update_layout(legend=dict(
            yanchor="top", y=legend_y, xanchor="right", x=legend_x,
            bordercolor="darkgrey", borderwidth=1, font=dict(size=15)
        ))

    fig.update_xaxes(
        title_text="Layer", title_standoff=4,
        mirror=True, showgrid=True, gridcolor='darkgrey',
        zeroline = True, zerolinecolor='darkgrey',
        title_font=dict(size=16), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey',
        range=x_range
    )
    fig.update_yaxes(
        title_text=y_title, title_standoff=4,
        mirror=True, showgrid=True, gridcolor='darkgrey',
        zeroline = True, zerolinecolor='darkgrey',
        title_font=dict(size=16), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey',
        range=y_range
    )
    fig.show()
    return fig