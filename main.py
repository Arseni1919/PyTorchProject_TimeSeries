import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

x = np.linspace(0, 2 * np.pi, 100)
sin_y = np.sin(x)
cos_y = np.cos(x)

trace1 = go.Scatter(x=x, y=sin_y, mode="lines", name="sin(x)")
trace2 = go.Scatter(x=x, y=cos_y, mode="lines", name="cos(x)")

data = [trace1, trace2]
layout = go.Layout(
    title="Sine and Cosine Plot", xaxis=dict(title="x"), yaxis=dict(title="y")
)
fig = go.Figure(data=data, layout=layout)

pyo.plot(fig, filename="sin_cos_plot.html")
