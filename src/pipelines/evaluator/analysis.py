import pandas as pd
import plotly.express as px


class EvaluatorAnalysis:
    @classmethod
    def plot_bar_chart(cls, counts_dataframe: pd.DataFrame, x_label: str):
        fig = px.bar(
            counts_dataframe, 
            x=x_label, 
            y="count", 
            title="Category Count",
            labels={x_label: "Category", "count": "Count"},
            text="count",
            color="count", 
            color_continuous_scale="plasma",
            height=800, 
            width=1200
        )
        
        fig.update_traces(
            textposition='outside'
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            template='plotly_dark', 
            title=dict(
                text="Category Count Distribution",
                font=dict(size=18, color="white"),
                x=0.5
            ),
            xaxis=dict(
                title=dict(font=dict(size=14, color="white")),
                tickfont=dict(size=12, color="white"),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(font=dict(size=14, color="white")),
                tickfont=dict(size=12, color="white"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.2)"
            ),
            coloraxis_colorbar=dict(
                title="Count",
                tickfont=dict(color="white")
            ),
            hoverlabel=dict(
                bgcolor="black",
                font_size=12,
                font_color="white"
            )
        )
        
        return fig