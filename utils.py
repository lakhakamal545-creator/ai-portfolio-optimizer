import numpy as np

def calculate_portfolio_health(sharpe, max_dd, div_score):
    score = 50
    if sharpe > 1.0: score += 20
    elif sharpe > 0.5: score += 10
    if max_dd > -0.15: score += 15
    elif max_dd > -0.30: score += 5
    if div_score < 0.5: score += 15
    return min(100, max(0, score))

def get_ai_recommendation(health_score, sharpe):
    if health_score >= 80: return "Excellent Risk-Adjusted Returns. The portfolio is highly optimized and well-diversified."
    elif health_score >= 60: return "Solid Portfolio. Consider adding uncorrelated assets to improve the diversification score and reduce maximum drawdown."
    else: return "Suboptimal Allocation. High concentration risk or severe drawdowns detected. Re-evaluating the weighting strategy via Minimum Variance is recommended."

def calculate_diversification_score(corr_matrix):
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_corr = corr_matrix.where(mask).mean().mean()
    return avg_corr
