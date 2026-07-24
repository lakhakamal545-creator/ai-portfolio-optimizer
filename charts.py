import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

BG_COLOR = "#0B132B"
PAPER_COLOR = "#1C2541"
TEXT_COLOR = "#E0E1DD"
ACCENT_GOLD = "#D4AF37"
ACCENT_BLUE = "#3A506B"
COLORS = ['#D4AF37', '#5BC0BE', '#3A506B', '#1C2541', '#E0E1DD', '#FF9F1C', '#E71D36']

def apply_premium_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=ACCENT_GOLD, size=20, family="Inter, sans-serif")),
        plot_bgcolor=PAPER_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#2b3a5a', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#2b3a5a', zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_COLOR)),
        hoverlabel=dict(bgcolor=PAPER_COLOR, font_size=13, font_family="Inter")
    )
    return fig

def plot_efficient_frontier(random_results, max_sharpe, min_vol, eff_returns, eff_vols):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=random_results[0,:], y=random_results[1,:], mode='markers', marker=dict(color=random_results[2,:], colorscale='Viridis', showscale=True, size=5, colorbar=dict(title='Sharpe Ratio')), name='Random Portfolios'))
    fig.add_trace(go.Scatter(x=eff_vols, y=eff_returns, mode='lines', line=dict(color='white', width=2, dash='dash'), name='Efficient Frontier'))
    fig.add_trace(go.Scatter(x=[max_sharpe['volatility']], y=[max_sharpe['return']], mode='markers+text', marker=dict(color='red', size=12, symbol='star'), name='Max Sharpe', text=['Max Sharpe'], textposition="top center"))
    fig.add_trace(go.Scatter(x=[min_vol['volatility']], y=[min_vol['return']], mode='markers+text', marker=dict(color='blue', size=12, symbol='star'), name='Min Volatility', text=['Min Volatility'], textposition="bottom center"))
    apply_premium_layout(fig, "Efficient Frontier & Capital Market Line")
    fig.update_xaxes(title="Annualized Volatility (Risk)")
    fig.update_yaxes(title="Annualized Return")
    return fig

def plot_allocation_pie(weights, tickers):
    df = pd.DataFrame({'Asset': tickers, 'Weight': weights})
    df = df[df['Weight'] > 0.01]
    fig = px.pie(df, values='Weight', names='Asset', hole=0.4, color_discrete_sequence=COLORS)
    apply_premium_layout(fig, "Portfolio Allocation")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_cumulative_returns(port_returns, bench_returns):
    port_cum = (1 + port_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, mode='lines', name='Portfolio', line=dict(color=ACCENT_GOLD, width=2)))
    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, mode='lines', name='Benchmark', line=dict(color='#5BC0BE', width=2)))
    apply_premium_layout(fig, "Cumulative Returns Comparison")
    fig.update_yaxes(title="Growth of $1")
    return fig

def plot_drawdown(drawdowns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, fill='tozeroy', mode='lines', line=dict(color='#E71D36', width=1), fillcolor='rgba(231, 29, 54, 0.3)', name='Drawdown'))
    apply_premium_layout(fig, "Portfolio Drawdown")
    fig.update_yaxes(title="Drawdown (%)", tickformat='.1%')
    return fig

def plot_correlation_heatmap(returns):
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1))
    apply_premium_layout(fig, "Asset Correlation Heatmap")
    return fig

def plot_monte_carlo_dist(final_values):
    fig = px.histogram(final_values, nbins=50, color_discrete_sequence=[ACCENT_GOLD])
    apply_premium_layout(fig, "Monte Carlo Simulation: 1 Year Future Value Distribution")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Expected Portfolio Value")
    fig.update_yaxes(title="Frequency")
    return fig
