import plotly.graph_objects as go

def create_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA20"],
            mode="lines",
            name="EMA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA50"],
            mode="lines",
            name="EMA50"
        )
    )

    fig.update_layout(
        title="NIFTY Live Chart",
        xaxis_rangeslider_visible=False,
        height=650
    )

    return fig