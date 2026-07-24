,import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

from data import fetch_data, fetch_benchmark
from optimizer import max_sharpe_ratio, min_variance, generate_random_portfolios, efficient_frontier
from risk import calculate_drawdowns, calculate_var_cvar, calculate_beta_alpha, calculate_sortino_treynor
from charts import plot_efficient_frontier, plot_allocation_pie, plot_cumulative_returns, plot_drawdown, plot_correlation_heatmap, plot_monte_carlo_dist
from utils import calculate_portfolio_health, get_ai_recommendation, calculate_diversification_score
from report import generate_pdf_report

st.set_page_config(page_title="Kamal Lakha | AI Portfolio Optimizer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown('''
    <style>
    :root { --bg-color: #0B132B; --card-bg: #1C2541; --accent-gold: #D4AF37; --text-color: #E0E1DD; }
    .stApp { background-color: var(--bg-color); color: var(--text-color); }
    div[data-testid="stSidebar"] { background-color: var(--card-bg); border-right: 1px solid #3A506B; }
    .css-1d391kg { background-color: var(--card-bg); }
    h1, h2, h3 { color: var(--accent-gold) !important; font-family: 'Playfair Display', serif; }
    .metric-card { background: var(--card-bg); padding: 20px; border-radius: 10px; border-top: 3px solid var(--accent-gold); text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-value { font-size: 28px; font-weight: bold; color: #fff; }
    .metric-title { font-size: 14px; color: #8D99AE; text-transform: uppercase; letter-spacing: 1px; }
    .health-score { font-size: 48px; font-weight: bold; color: #5BC0BE; }
    </style>
''', unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/BlackRock_logo.svg/512px-BlackRock_logo.svg.png", width=150)
st.sidebar.markdown("### AI Portfolio Optimization")
page = st.sidebar.radio("Navigation", ["Optimizer & MPT", "Risk Analytics", "Monte Carlo Simulation", "Correlation Analysis", "Performance Dashboard", "Export Report"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS")
benchmark_input = st.sidebar.text_input("Benchmark Ticker", "^NSEI")

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365*3))
end_date = col2.date_input("End Date", datetime.date.today())
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=7.0, step=0.1) / 100
initial_investment = st.sidebar.number_input("Investment Amount (₹)", min_value=1000, value=1000000, step=10000)

opt_objective = st.sidebar.selectbox("Optimization Objective", ["Maximum Sharpe Ratio", "Minimum Variance"])

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; font-size: 12px; color: #8D99AE;'>Developed by Kamal Lakha<br>MBA Financial Markets</p>", unsafe_allow_html=True)

@st.cache_data
def get_data(tickers, start, end): return fetch_data(tickers, start, end)

@st.cache_data
def get_bench(start, end, t): return fetch_benchmark(start, end, t)

try:
    with st.spinner("Fetching Market Data..."):
        prices, returns = get_data(tickers_input, start_date, end_date)
        bench_returns = get_bench(start_date, end_date, benchmark_input)
        
    tickers = list(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    if opt_objective == "Maximum Sharpe Ratio":
        res = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    else:
        res = min_variance(mean_returns, cov_matrix)
        
    opt_weights = res.x
    opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    opt_ret = np.sum(mean_returns * opt_weights)
    opt_sharpe = (opt_ret - risk_free_rate) / opt_vol
    port_returns = (returns * opt_weights).sum(axis=1)

    if page == "Optimizer & MPT":
        st.title("Portfolio Optimizer & Modern Portfolio Theory")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-title">Expected Return</div><div class="metric-value">{opt_ret:.2%}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">Annual Volatility</div><div class="metric-value">{opt_vol:.2%}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Sharpe Ratio</div><div class="metric-value">{opt_sharpe:.2f}</div></div>', unsafe_allow_html=True)
        drawdowns, max_dd = calculate_drawdowns(port_returns)
        c4.markdown(f'<div class="metric-card"><div class="metric-title">Max Drawdown</div><div class="metric-value" style="color:#E71D36;">{max_dd:.2%}</div></div>', unsafe_allow_html=True)
        st.write("---")
        
        col_mpt1, col_mpt2 = st.columns([2, 1])
        with col_mpt1:
            st.subheader("Efficient Frontier")
            with st.spinner("Simulating Portfolios..."):
                random_res, _ = generate_random_portfolios(2000, mean_returns, cov_matrix, risk_free_rate)
                target_returns = np.linspace(random_res[1,:].min(), random_res[1,:].max(), 50)
                ms = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
                mv = min_variance(mean_returns, cov_matrix)
                ms_vol, ms_ret = np.sqrt(np.dot(ms.x.T, np.dot(cov_matrix, ms.x))), np.sum(mean_returns * ms.x)
                mv_vol, mv_ret = np.sqrt(np.dot(mv.x.T, np.dot(cov_matrix, mv.x))), np.sum(mean_returns * mv.x)
                fig = plot_efficient_frontier(random_res, {'volatility': ms_vol, 'return': ms_ret}, {'volatility': mv_vol, 'return': mv_ret}, [mv_ret, ms_ret], [mv_vol, ms_vol])
                st.plotly_chart(fig, use_container_width=True)
                
        with col_mpt2:
            st.subheader("Optimal Allocation")
            fig_pie = plot_allocation_pie(opt_weights, tickers)
            st.plotly_chart(fig_pie, use_container_width=True)
            div_score = calculate_diversification_score(returns.corr())
            h_score = calculate_portfolio_health(opt_sharpe, max_dd, div_score)
            st.markdown(f"<div style='text-align:center;'><h4>Portfolio Health Score</h4><span class='health-score'>{h_score:.0f}/100</span></div>", unsafe_allow_html=True)
            st.info(f"**AI Recommendation:** {get_ai_recommendation(h_score, opt_sharpe)}")

    elif page == "Risk Analytics":
        st.title("Advanced Risk Analytics")
        beta, alpha = calculate_beta_alpha(port_returns, bench_returns, risk_free_rate)
        sortino, treynor = calculate_sortino_treynor(port_returns, beta, risk_free_rate)
        var_h, var_p, cvar = calculate_var_cvar(port_returns)
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f'<div class="metric-card"><div class="metric-title">Portfolio Beta (vs {benchmark_input})</div><div class="metric-value">{beta:.2f}</div></div>', unsafe_allow_html=True)
        rc2.markdown(f'<div class="metric-card"><div class="metric-title">Jensen\'s Alpha</div><div class="metric-value">{alpha:.2%}</div></div>', unsafe_allow_html=True)
        rc3.markdown(f'<div class="metric-card"><div class="metric-title">Sortino Ratio</div><div class="metric-value">{sortino:.2f}</div></div>', unsafe_allow_html=True)
        st.write("---")
        st.subheader("Value at Risk (VaR) & Expected Shortfall")
        v1, v2, v3 = st.columns(3)
        v1.metric("Historical Daily VaR", f"{var_h:.2%}")
        v2.metric("Parametric Daily VaR", f"{var_p:.2%}")
        v3.metric("Conditional VaR (Expected Shortfall)", f"{cvar:.2%}")
        st.write("---")
        st.subheader("Historical Drawdown Profile")
        drawdowns, _ = calculate_drawdowns(port_returns)
        fig_dd = plot_drawdown(drawdowns)
        st.plotly_chart(fig_dd, use_container_width=True)

    elif page == "Monte Carlo Simulation":
        st.title("Monte Carlo Portfolio Simulation")
        days, sims = 252, 10000
        daily_vol, daily_ret = opt_vol / np.sqrt(252), opt_ret / 252
        Z = np.random.normal(0, 1, (sims, days))
        sim_returns = daily_ret - 0.5 * daily_vol**2 + daily_vol * Z
        sim_paths = initial_investment * np.exp(np.cumsum(sim_returns, axis=1))
        final_values = sim_paths[:, -1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Value (Mean)", f"₹{np.mean(final_values):,.2f}")
        c2.metric("5th Percentile (Worst Case)", f"₹{np.percentile(final_values, 5):,.2f}")
        c3.metric("95th Percentile (Best Case)", f"₹{np.percentile(final_values, 95):,.2f}")
        fig_mc = plot_monte_carlo_dist(final_values)
        st.plotly_chart(fig_mc, use_container_width=True)

    elif page == "Correlation Analysis":
        st.title("Asset Correlation & Diversification")
        div_score = calculate_diversification_score(returns.corr())
        st.markdown(f"**Average Off-Diagonal Correlation:** {div_score:.2f} (Lower is better for diversification)")
        fig_corr = plot_correlation_heatmap(returns)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.subheader("Covariance Matrix (Annualized)")
        st.dataframe(cov_matrix.style.background_gradient(cmap='Blues'))

    elif page == "Performance Dashboard":
        st.title("Historical Performance Tracking")
        fig_cum = plot_cumulative_returns(port_returns, bench_returns)
        st.plotly_chart(fig_cum, use_container_width=True)
        st.subheader("Performance Summary")
        bench_ret, bench_vol = bench_returns.mean() * 252, bench_returns.std() * np.sqrt(252)
        bench_sharpe = (bench_ret - risk_free_rate) / bench_vol
        perf_df = pd.DataFrame({"Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio"], "Optimized Portfolio": [f"{opt_ret:.2%}", f"{opt_vol:.2%}", f"{opt_sharpe:.2f}"], f"Benchmark ({benchmark_input})": [f"{bench_ret:.2%}", f"{bench_vol:.2%}", f"{bench_sharpe:.2f}"]})
        st.table(perf_df.set_index("Metric"))

    elif page == "Export Report":
        st.title("Automated Portfolio Reporting")
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Compiling PDF..."):
                drawdowns, max_dd = calculate_drawdowns(port_returns)
                filename = generate_pdf_report(tickers, opt_weights, opt_sharpe, opt_ret, opt_vol, max_dd)
                with open(filename, "rb") as pdf_file: PDFbyte = pdf_file.read()
                st.download_button(label="Download Portfolio Report", data=PDFbyte, file_name="Kamal_Lakha_Portfolio_Report.pdf", mime='application/octet-stream')
                st.success("Report Generated Successfully!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}. Please check your ticker symbols or date range.")
