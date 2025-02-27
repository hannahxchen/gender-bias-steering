import plotly.graph_objects as go
from .plot_utils import COLORS


def plot_projection(projections, bias_scores, width=350, height=300, x_range=None, y_range=None, title_text=None, annotations=None, annotate_x=8, annotate_y=-0.9):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=projections, y=bias_scores, mode="markers", marker_color=COLORS['before'], showlegend=False))
    fig.add_shape(type="line", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="#66AA00", width=3, dash="dash"), layer="below"
    )
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18), x=0.55, y=0.97),
        plot_bgcolor='white', width=width, height=height, margin=dict(l=20, r=15, t=25, b=20), 
        font=dict(size=13), legend_title_font=dict(size=14), title_font=dict(size=14)
    )
    fig.update_yaxes(
        title_text="Disparity Score", title_standoff=4,
        title_font=dict(size=15), tickfont=dict(size=13),
        mirror=True, showgrid=True, gridcolor='darkgrey', 
        zeroline = True, zerolinecolor='darkgrey', 
        showline=True, linewidth=1, linecolor='darkgrey',
        range=y_range
    )
    fig.update_xaxes(
        title_text="Projection", title_standoff=5, 
        title_font=dict(size=15), tickfont=dict(size=13),
        mirror=True, showgrid=True, gridcolor='darkgrey', 
        zeroline = True, zerolinecolor='darkgrey', 
        showline=True, linewidth=1, linecolor='darkgrey',
        range=x_range
    )
    if annotations is not None:
        layer = annotations["layer"]
        corr = annotations["corr"]
        rmse = annotations["rmse"]
        fig.add_annotation(
            text=f'Layer={layer}<br>r={corr:.4f}<br>RMSE={rmse:.4f}', 
            font=dict(size=14), align='left',
            showarrow=False, bgcolor="white",
            x=annotate_x, y=annotate_y, bordercolor='black', borderwidth=1
        )
    fig.show()
    return fig



def plot_debias(
    bias_scores, baseline_bias, projections, width=360, height=300, 
    x_range=None, y_range=None, showlegend=True, title_text=None,
    legend_x=0.02, legend_y=0.98, opacity=1
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=projections, y=baseline_bias,  mode="markers", marker_color=COLORS['before'], marker_size=5, name="before", showlegend=showlegend))
    fig.add_trace(go.Scatter(x=projections, y=bias_scores, mode="markers", marker_color=COLORS['after'], marker_size=5, name="after", showlegend=showlegend, opacity=opacity))
    
    fig.update_layout(
        font=dict(size=14), plot_bgcolor='white', 
        title=dict(text=title_text, font=dict(size=17), x=0.55, y=0.98),
        margin=dict(l=20, r=15, t=25, b=20), width=width, height=height,
        legend=dict(
            yanchor="top", y=legend_y, xanchor="left", x=legend_x,
            bordercolor="darkgrey", borderwidth=1, font=dict(size=15)
        ),
    )
    fig.update_xaxes(
        title_text="Projection", title_standoff=5,
        mirror=True, showgrid=True, gridcolor='darkgrey',
        zeroline = True, zerolinecolor='darkgrey',
        title_font=dict(size=16), tickfont=dict(size=13),
        showline=True, linewidth=1, linecolor='darkgrey', range=x_range
    )
    fig.update_yaxes(
        title_text="Disparity Score",
        mirror=True, showgrid=True, gridcolor='darkgrey', 
        zeroline = True, zerolinecolor='darkgrey',
        title_font=dict(size=16), tickfont=dict(size=13), title_standoff=4,
        showline=True, linewidth=1, linecolor='darkgrey', range=y_range
    )
    fig.show()
    return fig


