import numpy as np
import pandas as pd
import scipy.stats as stats

def calculate_drawdowns(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    return drawdowns, max_drawdown

def calculate_var_cvar(returns, confidence_level=0.95):
    var_hist = np.percentile(returns, (1 - confidence_level) * 100)
    mu = np.mean(returns)
    std = np.std(returns)
    var_param = stats.norm.ppf(1 - confidence_level, mu, std)
    cvar = returns[returns <= var_hist].mean()
    return var_hist, var_param, cvar

def calculate_beta_alpha(port_returns, bench_returns, risk_free_rate=0.07):
    df = pd.concat([port_returns, bench_returns], axis=1).dropna()
    if df.empty or len(df.columns) < 2:
        return 1.0, 0.0
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    var = np.var(df.iloc[:, 1])
    beta = cov / var if var != 0 else 1.0
    ann_port_ret = np.mean(df.iloc[:, 0]) * 252
    ann_bench_ret = np.mean(df.iloc[:, 1]) * 252
    alpha = ann_port_ret - (risk_free_rate + beta * (ann_bench_ret - risk_free_rate))
    return beta, alpha

def calculate_sortino_treynor(returns, beta, risk_free_rate=0.07):
    ann_ret = np.mean(returns) * 252
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 1e-6
    sortino = (ann_ret - risk_free_rate) / downside_std
    treynor = (ann_ret - risk_free_rate) / beta if beta != 0 else 0
    return sortino, treynor
