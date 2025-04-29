import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from wordcloud import WordCloud
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import requests
import time

# --- Config ---
st.set_page_config(page_title="Argentis Investment", layout="wide")

# --- Configuration de la session pour yfinance ---
custom_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://finance.yahoo.com/'
}
session = requests.Session()
session.headers.update(custom_headers)

# --- Personnalisation CSS ---
CSS_STYLE = """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        color: #1f1f1f;
        margin: 0;
        padding: 0;
    }
    h1 {
        color: #003366;
        font-size: 48px;
        text-align: center;
        margin-top: 0;
        padding-top: 10px;
    }
    h2, h3, h4, .caption {
        color: #1a1a1a !important;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
    }
    section[data-testid="stSidebar"] {
        background-color: #e6f0ff;
        padding: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100vh;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 1.25rem;
        font-weight: bold;
        display: flex;
        justify-content: center;
        text-align: center;
        margin: 10px 0;
        color: #1a1a1a;
    }
    section[data-testid="stSidebar"] .stRadio {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 0 10px;
    }
    div.stButton > button, div.stDownloadButton > button {
        background-color: #1a75ff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #005ce6;
    }
    .news-ticker {
        background-color: #e6f0ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        overflow: hidden;
        white-space: nowrap;
        position: relative;
    }
    .news-ticker-container {
        display: inline-block;
        animation: scroll 50s linear infinite;
    }
    .news-item {
        display: inline-block;
        margin-right: 20px;
        font-size: 15px;
        color: #00008B;
    }
    @keyframes scroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .welcome-text {
        font-size: 1.125rem;
    }
    .custom-footer {
        background-color: #f5f5f5;
        color: #888;
        text-align: center;
        padding: 5px 0;
        font-size: 0.75rem;
        position: fixed;
        bottom: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- Header ---
st.markdown(
    f"""
    <div style='background-color:#ADD8E6;padding:10px;border-radius:10px;margin-bottom:20px'>
    <h1>💵 Argentis Investment 💵</h1>
    <h4>Votre plateforme financière VIP</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("Navigation", [
    "Accueil",
    "Évaluation d'un Actif",
    "Comparateur d'Actifs",
    "Gestion de Portefeuille",
    "Prévisions Machine Learning",
    "Sentiment & NLP",
    "ESG & Durabilité",
    "Recommandations Automatiques",
    "Gestion Avancée des Risques",
    "Optimisation de Portefeuille",
    "Dashboard Personnalisé",
    "Suivi Temps Réel du Portefeuille",
    "Export & Reporting"
])

# --- Utils ---
@st.cache_data
@retry(
    stop=stop_after_attempt(10),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(Exception)
)
def get_ticker_data(ticker_input):
    """Fetch all required data for a ticker using yfinance."""
    try:
        time.sleep(3)  # Délai pour éviter les blocages
        ticker = yf.Ticker(ticker_input, session=session)
        info = ticker.info
        historical_data = ticker.history(period="5y", interval="1d")
        sustainability = ticker.sustainability
        news = ticker.news

        if historical_data.empty or 'Close' not in historical_data.columns or len(historical_data) < 2:
            st.warning(f"Les données historiques pour {ticker_input} sont vides ou incomplètes.")
            historical_data = None

        return {
            "info": info,
            "historical_data": historical_data,
            "sustainability": sustainability,
            "news": news
        }
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {ticker_input} : {str(e)}")
        st.info("Cela peut être dû à une limitation de l'API de Yahoo Finance ou à un problème de réseau. Essayez un autre ticker (par exemple, AAPL ou MSFT) ou réessayez plus tard.")
        return None

@st.cache_data
@retry(
    stop=stop_after_attempt(10),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(Exception)
)
def get_history(ticker_input, period="5y", interval="1d"):
    """Fetch historical stock data for a given ticker using yfinance."""
    try:
        time.sleep(3)
        ticker = yf.Ticker(ticker_input, session=session)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty or "Close" not in df.columns or len(df) < 2:
            st.warning(f"Les données pour {ticker_input} sont vides ou incomplètes.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données historiques pour {ticker_input} : {str(e)}")
        st.info("Cela peut être dû à une limitation de l'API de Yahoo Finance. Essayez un autre ticker (par exemple, AAPL ou MSFT).")
        return None

@st.cache_data
@retry(
    stop=stop_after_attempt(10),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(Exception)
)
def get_ratios(ticker_input):
    """Fetch key financial ratios for a given ticker using yfinance."""
    try:
        time.sleep(3)
        ticker = yf.Ticker(ticker_input, session=session)
        info = ticker.info

        if not info:
            st.warning(f"Les ratios financiers pour {ticker_input} sont indisponibles via yfinance.")
            return None

        return {
            "PER": info.get("trailingPE", "N/A"),
            "PBR": info.get("priceToBook", "N/A"),
            "ROE": info.get("returnOnEquity", "N/A"),
            "ROA": info.get("returnOnAssets", "N/A"),
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "Current Ratio": info.get("currentRatio", "N/A"),
            "Quick Ratio": info.get("quickRatio", "N/A"),
            "Gross Margin": info.get("grossMargins", "N/A"),
            "Net Margin": info.get("profitMargins", "N/A"),
        }
    except Exception as e:
        st.error(f"Erreur lors de la récupération des ratios pour {ticker_input} : {str(e)}")
        st.info("Cela peut être dû à une limitation de l'API de Yahoo Finance. Essayez un autre ticker (par exemple, AAPL ou MSFT).")
        return None

@st.cache_data
def get_top_bottom_performers():
    """Fetch top 5 and bottom 5 daily performers (fictitious data for now)."""
    top_5 = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "Rendement (%)": [5.2, 4.8, 3.9, 3.1, 2.7],
        "Variation ($)": [8.50, 12.30, 6.20, 9.10, 15.40]
    })
    bottom_5 = pd.DataFrame({
        "Ticker": ["NFLX", "PYPL", "DIS", "WMT", "PEP"],
        "Rendement (%)": [-4.5, -3.8, -3.2, -2.9, -2.1],
        "Variation ($)": [-7.80, -5.40, -4.10, -3.60, -2.50]
    })
    return top_5, bottom_5

@st.cache_data
def calculate_wacc(ticker_input, ticker_data=None):
    """Calculate a simplified WACC for a given ticker."""
    try:
        if ticker_data is None:
            ticker_data = get_ticker_data(ticker_input)
        if ticker_data is None or ticker_data.get("info") is None:
            return None
        info = ticker_data["info"]
        total_debt = float(info.get("totalDebt", 0))
        market_cap = float(info.get("marketCap", 1))
        total_value = total_debt + market_cap
        debt_weight = total_debt / total_value if total_value > 0 else 0
        equity_weight = market_cap / total_value if total_value > 0 else 0
        cost_of_equity = 0.08
        cost_of_debt = 0.04
        tax_rate = 0.21
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        return wacc * 100
    except:
        return None

@st.cache_data
def calculate_dcf(ticker_input, ticker_data=None):
    """Calculate a simplified DCF valuation for a given ticker."""
    try:
        if ticker_data is None:
            ticker_data = get_ticker_data(ticker_input)
        if ticker_data is None or ticker_data.get("info") is None:
            return None
        info = ticker_data["info"]
        cash_flow = float(info.get("operatingCashFlow", 0))
        growth_rate = 0.02
        discount_rate = 0.08
        years = 5
        dcf = 0
        for i in range(1, years + 1):
            dcf += cash_flow * (1 + growth_rate) ** i / (1 + discount_rate) ** i
        terminal_value = cash_flow * (1 + growth_rate) ** (years + 1) / (discount_rate - growth_rate)
        dcf += terminal_value / (1 + discount_rate) ** years
        return dcf / 1e6
    except:
        return None

# --- Page: Accueil ---
if page == "Accueil":
    st.title("🏠 Accueil")
    
    # Descriptif de l'application
    st.markdown(
        """
        <div class="welcome-text">
        Argentis Investment : Votre allié financier pour analyser, prévoir et optimiser vos investissements avec l'IA.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Défilement des infos
    st.subheader("📰 Actualités Financières")
    news_items = [
        {"title": "Les marchés atteignent un nouveau record cette semaine !", "icon": "📈"},
        {"title": "La tech continue de dominer avec de nouvelles innovations.", "icon": "💻"},
        {"title": "Investissements verts : une tendance qui s’accélère.", "icon": "🌍"},
        {"title": "Volatilité accrue sur les marchés émergents.", "icon": "📉"},
        {"title": "Les banques centrales ajustent leurs taux directeurs.", "icon": "🏦"},
    ]
    repeated_news = news_items * 5
    news_html = '<div class="news-ticker"><div class="news-ticker-container">'
    for n in repeated_news:
        title = n["title"]
        icon = n["icon"]
        news_html += f'<span class="news-item">{icon} {title}</span>'
    news_html += '</div></div>'
    st.markdown(news_html, unsafe_allow_html=True)
    
    # Top 5 hausses et baisses
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Top 5 Hausses d'aujourd'hui")
        top_5, bottom_5 = get_top_bottom_performers()
        if not top_5.empty:
            top_5["Rendement (%)"] = top_5["Rendement (%)"].apply(lambda x: f"{x:.2f}")
            top_5["Variation ($)"] = top_5["Variation ($)"].apply(lambda x: f"{x:.2f}")
            st.table(top_5.style.applymap(lambda x: "color: #28a745", subset=["Rendement (%)", "Variation ($)"]))
        else:
            st.write("Aucune donnée disponible pour les hausses.")
    
    with col2:
        st.subheader("📉 Top 5 Baisses d'aujourd'hui")
        if not bottom_5.empty:
            bottom_5["Rendement (%)"] = bottom_5["Rendement (%)"].apply(lambda x: f"{x:.2f}")
            bottom_5["Variation ($)"] = bottom_5["Variation ($)"].apply(lambda x: f"{x:.2f}")
            st.table(bottom_5.style.applymap(lambda x: "color: #dc3545", subset=["Rendement (%)", "Variation ($)"]))
        else:
            st.write("Aucune donnée disponible pour les baisses.")
    
    # Recherche sur un titre ou une compagnie
    st.subheader("🔍 Recherche sur un titre ou une compagnie")
    ticker_input = st.text_input("Symbole ou compagnie", key="home_ticker")
    if ticker_input:
        ticker_data = get_ticker_data(ticker_input)
        if ticker_data is None:
            st.error(f"Impossible de récupérer les données pour {ticker_input}. Essayez un autre ticker (par exemple, AAPL ou MSFT).")
        else:
            info = ticker_data.get("info")
            data = ticker_data.get("historical_data")
            
            if data is None or data.empty:
                st.error(f"Aucune donnée historique disponible pour {ticker_input}.")
            else:
                if "Close" not in data.columns:
                    st.error(f"Les données historiques pour {ticker_input} ne contiennent pas la colonne 'Close'.")
                else:
                    price = data["Close"].iloc[-1]
                    st.metric(label=f"Prix actuel de {ticker_input}", value=f"{float(price):.2f} $")
                    
                    period_options = {
                        "1 jour": "1d",
                        "5 jours": "5d",
                        "1 mois": "1mo",
                        "6 mois": "6mo",
                        "1 an": "1y",
                        "5 ans": "5y"
                    }
                    selected_period = st.selectbox("Choisir l'horizon temporel", options=list(period_options.keys()))
                    period = period_options[selected_period]
                    
                    historical_data = get_history(ticker_input, period=period)
                    if historical_data is not None and not historical_data.empty and "Close" in historical_data.columns:
                        st.line_chart(historical_data["Close"])
                    else:
                        st.write(f"Aucune donnée disponible pour afficher le graphique de {ticker_input} sur la période sélectionnée.")
                    
                    st.subheader("Informations sur le titre ou la compagnie")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        market_cap = info.get("marketCap", "N/A")
                        if market_cap != "N/A":
                            market_cap = f"{float(market_cap) / 1e9:.2f} Mds $"
                        st.metric("Capitalisation boursière", market_cap)
                        
                        bid = info.get("bid", "N/A")
                        if bid != "N/A":
                            bid = f"{float(bid):.2f} $"
                        st.metric("Offre", bid)
                        
                        ask = info.get("ask", "N/A")
                        if ask != "N/A":
                            ask = f"{float(ask):.2f} $"
                        st.metric("Demande", ask)
                    
                    with col2:
                        week_low = info.get("fiftyTwoWeekLow", "N/A")
                        week_high = info.get("fiftyTwoWeekHigh", "N/A")
                        week_range = f"{float(week_low):.2f} - {float(week_high):.2f} $" if week_low != "N/A" and week_high != "N/A" else "N/A"
                        st.metric("Plage sur 52 semaines", week_range)
                        
                        pe_ratio = info.get("trailingPE", "N/A")
                        if pe_ratio != "N/A":
                            pe_ratio = f"{float(pe_ratio):.2f}"
                        st.metric("Ratio P/E", pe_ratio)
                        
                        dividend_yield = info.get("dividendYield", "N/A")
                        if dividend_yield != "N/A":
                            dividend_yield = f"{float(dividend_yield) * 100:.2f}%"
                        st.metric("Rendement du dividende", dividend_yield)

# --- Page: Évaluation d'un Actif ---
elif page == "Évaluation d'un Actif":
    st.title("📈 Évaluation d'un Actif")
    ticker_input = st.text_input("Symbole ou compagnie à évaluer", key="eval")
    
    if ticker_input:
        ticker_data = get_ticker_data(ticker_input)
        
        st.subheader("WACC (Coût Moyen Pondéré du Capital)")
        if ticker_data:
            wacc = calculate_wacc(ticker_input, ticker_data)
            if wacc is not None:
                st.metric("WACC", f"{wacc:.2f}%")
            else:
                st.write("Données insuffisantes pour calculer le WACC.")
        else:
            st.write("Impossible de récupérer les données pour calculer le WACC.")
    
        st.subheader("DCF (Valorisation par Flux de Trésorerie Actualisés)")
        if ticker_data:
            dcf = calculate_dcf(ticker_input, ticker_data)
            if dcf is not None:
                st.metric("DCF", f"{dcf:.2f} M$")
            else:
                st.write("Données insuffisantes pour calculer le DCF.")
        else:
            st.write("Impossible de récupérer les données pour calculer le DCF.")
    
        st.subheader("Ratios Financiers")
        if ticker_data:
            ratios = get_ratios(ticker_input)
            if ratios:
                st.table(pd.DataFrame.from_dict(ratios, orient='index', columns=['Valeur']))
            else:
                st.write("Aucune donnée disponible pour les ratios financiers.")
        else:
            st.write("Impossible de récupérer les données pour les ratios financiers.")
    
        st.subheader("Historique des Prix")
        if ticker_data and ticker_data.get("historical_data") is not None:
            df = ticker_data["historical_data"]
            if not df.empty and "Close" in df.columns:
                st.line_chart(df["Close"])
            else:
                st.write("Aucune donnée disponible pour afficher l'historique des prix.")
        else:
            st.write("Impossible de récupérer les données historiques.")

# --- Page: Comparateur d'Actifs ---
elif page == "Comparateur d'Actifs":
    st.title("🔎 Comparateur d'Actifs")
    col1, col2 = st.columns(2)
    with col1:
        ticker_1 = st.text_input("Actif 1", key="c1", placeholder="Ex: AAPL")
    with col2:
        ticker_2 = st.text_input("Actif 2", key="c2", placeholder="Ex: MSFT")

    if ticker_1 and ticker_2:
        ticker_data_1 = get_ticker_data(ticker_1)
        ticker_data_2 = get_ticker_data(ticker_2)
        
        if ticker_data_1 is None or ticker_data_2 is None:
            st.error(f"Impossible de récupérer les données pour {ticker_1} ou {ticker_2}. Essayez d'autres tickers (par exemple, AAPL et MSFT).")
        else:
            data_1 = ticker_data_1.get("historical_data")
            data_2 = ticker_data_2.get("historical_data")
            
            if data_1 is None or data_2 is None or data_1.empty or data_2.empty:
                st.error(f"Aucune donnée historique disponible pour {ticker_1} ou {ticker_2}.")
            else:
                st.subheader("Comparaison des Ratios Financiers")
                ratios_1 = get_ratios(ticker_1)
                ratios_2 = get_ratios(ticker_2)
                if ratios_1 and ratios_2:
                    comp_df = pd.DataFrame({ticker_1: ratios_1, ticker_2: ratios_2})
                    comp_df = comp_df.replace("N/A", "-")
                    st.table(comp_df)
                else:
                    st.write(f"Aucune donnée disponible pour comparer les ratios financiers de {ticker_1} ou {ticker_2}.")

                st.subheader("Comparaison des Performances")
                if "Close" in data_1.columns and "Close" in data_2.columns:
                    rets_1 = data_1["Close"].pct_change().dropna()
                    rets_2 = data_2["Close"].pct_change().dropna()

                    if len(rets_1) < 2 or len(rets_2) < 2:
                        st.write(f"Données insuffisantes pour calculer les performances de {ticker_1} ou {ticker_2}.")
                    else:
                        return_1 = rets_1.mean() * 252 * 100
                        return_2 = rets_2.mean() * 252 * 100
                        vol_1 = rets_1.std() * np.sqrt(252) * 100
                        vol_2 = rets_2.std() * np.sqrt(252) * 100

                        return_1 = return_1.iloc[0] if isinstance(return_1, pd.Series) else float(return_1)
                        return_2 = return_2.iloc[0] if isinstance(return_2, pd.Series) else float(return_2)
                        vol_1 = vol_1.iloc[0] if isinstance(vol_1, pd.Series) else float(vol_1)
                        vol_2 = vol_2.iloc[0] if isinstance(vol_2, pd.Series) else float(vol_2)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"Rendement Annualisé ({ticker_1})", f"{return_1:.2f}%")
                            st.metric(f"Volatilité Annualisée ({ticker_1})", f"{vol_1:.2f}%")
                        with col2:
                            st.metric(f"Rendement Annualisé ({ticker_2})", f"{return_2:.2f}%")
                            st.metric(f"Volatilité Annualisée ({ticker_2})", f"{vol_2:.2f}%")
                else:
                    st.write(f"Données insuffisantes pour calculer les performances de {ticker_1} ou {ticker_2}.")

                st.subheader("Performance Historique")
                period_options = {
                    "1 jour": "1d",
                    "5 jours": "5d",
                    "1 mois": "1mo",
                    "6 mois": "6mo",
                    "1 an": "1y",
                    "5 ans": "5y"
                }
                selected_period = st.selectbox("Choisir l'horizon temporel", options=list(period_options.keys()), key="comp_period")
                period = period_options[selected_period]

                df1 = get_history(ticker_1, period=period)
                df2 = get_history(ticker_2, period=period)

                if df1 is not None and df2 is not None and not df1.empty and not df2.empty and "Close" in df1.columns and "Close" in df2.columns:
                    if len(df1["Close"]) < 2 or len(df2["Close"]) < 2:
                        st.write(f"Données insuffisantes pour afficher la performance historique.")
                    else:
                        close_1 = df1["Close"]
                        close_2 = df2["Close"]
                        if not isinstance(close_1, pd.Series):
                            close_1 = pd.Series([close_1], index=[df1.index[0]])
                        if not isinstance(close_2, pd.Series):
                            close_2 = pd.Series([close_2], index=[df2.index[0]])
                        combined = pd.DataFrame({
                            ticker_1: close_1,
                            ticker_2: close_2
                        }).dropna()
                        if not combined.empty and len(combined) >= 2:
                            st.line_chart(combined)
                        else:
                            st.write(f"Données insuffisantes pour afficher la performance historique.")
                else:
                    st.write(f"Aucune donnée disponible pour afficher la performance historique.")
    else:
        st.write("Veuillez saisir les deux actifs pour lancer la comparaison.")

# --- Page: Gestion de Portefeuille ---
elif page == "Gestion de Portefeuille":
    st.title("📚 Gestion de Portefeuille")
    st.write("Saisissez jusqu'à 10 titres avec leurs poids (en %). La somme des poids doit être proche de 100%.")
    
    st.subheader("Saisie du Portefeuille")
    col1, col2 = st.columns(2)
    portfolio = {}
    with col1:
        for i in range(1, 6):
            ticker = st.text_input(f"Symbole ou compagnie {i}", key=f"ticker_{i}", placeholder="Ex: AAPL")
            weight = st.number_input(f"Poids (%) {i}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"weight_{i}")
            if ticker and weight > 0:
                portfolio[ticker] = weight / 100
    with col2:
        for i in range(6, 11):
            ticker = st.text_input(f"Symbole ou compagnie {i}", key=f"ticker_{i}", placeholder="Ex: MSFT")
            weight = st.number_input(f"Poids (%) {i}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"weight_{i}")
            if ticker and weight > 0:
                portfolio[ticker] = weight / 100
    
    total_weight = sum(portfolio.values())
    if total_weight > 0 and not 0.95 <= total_weight <= 1.05:
        st.warning("La somme des poids doit être proche de 100% (entre 95% et 105%).")
    
    if st.button("Simuler Portefeuille"):
        if not portfolio:
            st.error("Veuillez saisir au moins un symbole ou compagnie avec un poids valide.")
        else:
            try:
                dfp = pd.DataFrame({t: get_history(t)["Close"] for t in portfolio})
                rets = dfp.pct_change().dropna()
                weights = np.array([portfolio[t] for t in portfolio])
                
                portfolio_return = np.sum(rets.mean() * weights) * 252 * 100
                st.metric("Rendement Annualisé", f"{portfolio_return:.2f}%")
                
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))) * 100
                st.metric("Volatilité Annualisée", f"{portfolio_vol:.2f}%")
                
                n = len(portfolio)
                sims = 5000
                results = np.zeros((3, sims))
                weights_list = []
                for i in range(sims):
                    w = np.random.dirichlet(np.ones(n))
                    weights_list.append(w)
                    ret = np.sum(rets.mean() * w) * 252
                    vol = np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))
                    results[:, i] = [vol, ret, ret / vol]
                idx = np.argmax(results[2])
                w_opt = weights_list[idx]
                df_opt = pd.DataFrame({'Actif': list(portfolio), 'Poids optimal': [f"{w * 100:.2f}%" for w in w_opt]})
                st.table(df_opt)
                
                fig, ax = plt.subplots()
                ax.scatter(results[0], results[1], c=results[2], cmap='viridis')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur lors de la simulation : {str(e)}")

# --- Page: Prévisions Machine Learning ---
elif page == "Prévisions Machine Learning":
    st.title("🤖 Prévisions Machine Learning")
    
    # Saisie des paramètres
    ticker_input = st.text_input("Symbole ou compagnie ML", key="ml", placeholder="Ex: AAPL")
    days = st.number_input("Jours à prévoir", 1, 30, 7, step=1)

    if ticker_input:
        # Récupération des données historiques
        ticker_data = get_ticker_data(ticker_input)
        if ticker_data is None or ticker_data.get("historical_data") is None:
            st.warning(f"Aucune donnée disponible pour {ticker_input}. Vérifiez le ticker ou essayez un autre symbole (par exemple, AAPL ou MSFT).")
        else:
            data = ticker_data["historical_data"]
            if data.empty or "Close" not in data.columns:
                st.warning(f"Aucune donnée historique valide pour {ticker_input}.")
            else:
                # Préparation des données pour les modèles
                df = data["Close"].reset_index()
                df.columns = ["ds", "y"]

                # Colonnes pour organiser les prévisions
                col1, col2 = st.columns(2)

                # Prévision ARIMA
                with col1:
                    st.subheader("Prévision ARIMA")
                    try:
                        model_arima = ARIMA(df["y"], order=(1, 1, 1))
                        arima_fit = model_arima.fit()
                        forecast_arima = arima_fit.forecast(steps=days)
                        forecast_values = forecast_arima.values
                        st.metric("Valeur Prédite (Dernier Jour)", f"{float(forecast_values[-1]):.2f} $")
                    except Exception as e:
                        st.warning(f"Erreur lors de la prévision ARIMA : {str(e)}. Essayez un autre ticker ou ajustez les paramètres.")

                # Prévision Prophet
                with col2:
                    st.subheader("Prévision Prophet")
                    try:
                        m = Prophet()
                        m.fit(df)
                        future = m.make_future_dataframe(periods=days)
                        forecast_prophet = m.predict(future)
                        pred_values = forecast_prophet["yhat"].tail(days).values
                        st.metric("Valeur Prédite (Dernier Jour)", f"{float(pred_values[-1]):.2f} $")
                    except Exception as e:
                        st.warning(f"Erreur lors de la prévision Prophet : {str(e)}. Essayez un autre ticker ou ajustez les paramètres.")

                # Graphique combiné des prévisions
                st.subheader("Graphique des Prévisions")
                try:
                    historical = df.set_index("ds")["y"]
                    forecast_dates = pd.date_range(start=df["ds"].iloc[-1] + pd.Timedelta(days=1), periods=days, freq="D")
                    forecast_arima_df = pd.DataFrame(forecast_values, index=forecast_dates, columns=["ARIMA"])
                    forecast_prophet_df = forecast_prophet[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
                    forecast_prophet_df = forecast_prophet_df.tail(days)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(historical.index, historical.values, label="Historique", color="#1a75ff")
                    ax.plot(forecast_arima_df.index, forecast_arima_df["ARIMA"], label="Prévision ARIMA", color="#28a745", linestyle="--")
                    ax.plot(forecast_prophet_df.index, forecast_prophet_df["yhat"], label="Prévision Prophet", color="#ff9900", linestyle="--")
                    ax.fill_between(
                        forecast_prophet_df.index,
                        forecast_prophet_df["yhat_lower"],
                        forecast_prophet_df["yhat_upper"],
                        color="#ff9900",
                        alpha=0.1,
                        label="Intervalle de Confiance (Prophet)"
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Prix de Clôture ($)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Erreur lors de l'affichage du graphique des prévisions : {str(e)}.")

# --- Page: Sentiment & NLP ---
elif page == "Sentiment & NLP":
    st.title("📰 Sentiment & NLP")
    
    ticker_sentiment = st.text_input("Symbole ou compagnie", key="sent", placeholder="Ex: AAPL")
    source = st.selectbox("Source des actualités", ["Données simulées", "X (simulé)"], index=0)
    
    if ticker_sentiment:
        ticker_data = get_ticker_data(ticker_sentiment)
        if ticker_data is None or ticker_data.get("news") is None:
            st.info("Les actualités ne sont pas disponibles pour ce ticker via yfinance. Utilisation de données simulées.")
            news = [
                {"title": f"{ticker_sentiment} annonce un nouveau produit innovant"},
                {"title": f"Les analystes prédisent une croissance pour {ticker_sentiment}"},
                {"title": f"{ticker_sentiment} fait face à une baisse des ventes"}
            ]
        else:
            news = ticker_data["news"]
            if not news:
                st.info("Aucune actualité récente disponible via yfinance. Utilisation de données simulées.")
                news = [
                    {"title": f"{ticker_sentiment} annonce un nouveau produit innovant"},
                    {"title": f"Les analystes prédisent une croissance pour {ticker_sentiment}"},
                    {"title": f"{ticker_sentiment} fait face à une baisse des ventes"}
                ]
        
        titles = [n.get("title", "") for n in news]
        
        sentiment_scores = []
        for t in titles:
            score = (t.lower().count('gain') + t.lower().count('strong') - 
                     t.lower().count('loss') - t.lower().count('weak'))
            if score > 0:
                sentiment_scores.append("Positif")
            elif score < 0:
                sentiment_scores.append("Négatif")
            else:
                sentiment_scores.append("Neutre")
        
        st.markdown("### Analyse des Sentiments")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribution des Sentiments")
            if sentiment_scores:
                sentiment_counts = pd.Series(sentiment_scores).value_counts()
                fig_sent, ax = plt.subplots()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["#28a745", "#ffc107", "#dc3545"])
                ax.set_ylabel("Nombre d'actualités")
                st.pyplot(fig_sent)
            else:
                st.write("Aucune donnée pour afficher la distribution des sentiments.")
        
        with col2:
            st.markdown("#### Nuage de Mots")
            text = " ".join(titles)
            if text.strip():
                wc = WordCloud(width=400, height=200, background_color='white', 
                               colormap='RdYlGn').generate(text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            else:
                st.write("Aucune actualité disponible pour générer un nuage de mots.")
        
        st.markdown("#### Actualités Récentes")
        if titles:
            for title, sent in zip(titles, sentiment_scores):
                color = {"Positif": "positive", "Négatif": "negative", "Neutre": ""}[sent]
                st.markdown(f"<span class='{color}'>- {title}</span>", unsafe_allow_html=True)
        else:
            st.write("Aucune actualité récente disponible.")

# --- Page: ESG & Durabilité ---
elif page == "ESG & Durabilité":
    st.title("🌿 ESG & Durabilité")
    ticker_esg = st.text_input("Symbole ou compagnie ESG", key="esg")

    if ticker_esg:
        ticker_data = get_ticker_data(ticker_esg)
        if ticker_data is None or ticker_data.get("sustainability") is None:
            st.warning(f"Les données ESG pour {ticker_esg} ne sont pas disponibles via yfinance.")
        else:
            sustainability = ticker_data["sustainability"]
            if sustainability is not None and not sustainability.empty:
                st.subheader("Données ESG")
                st.table(sustainability)
            else:
                st.warning(f"Les données ESG pour {ticker_esg} ne sont pas disponibles via yfinance.")

# --- Page: Recommandations Automatiques ---
elif page == "Recommandations Automatiques":
    st.title("⭐ Recommandations Automatiques")
    
    tickers_input = st.text_input("Symboles ou compagnies (séparés par des virgules)", key="reco", placeholder="Ex: AAPL,MSFT,GOOGL")
    sort_by = st.selectbox("Trier par", ["Score", "P/E", "Croissance"], index=0)
    
    if tickers_input:
        tl = [t.strip().upper() for t in tickers_input.split(',')]
        reco = {}
        for t in tl:
            ticker_data = get_ticker_data(t)
            if ticker_data and ticker_data.get("info"):
                info = ticker_data["info"]
                pe = info.get("trailingPE", "N/A")
                gr = info.get("earningsGrowth", "N/A")
                score = 0
                if pe != "N/A" and gr != "N/A":
                    score = 5 if pe < 20 and gr > 0 else (1 if pe > 30 else 3)
                reco[t] = {'P/E': pe, 'Croissance': gr, 'Score': score}
            else:
                st.warning(f"Impossible de récupérer les données pour {t}. Essayez un autre ticker.")
        
        if reco:
            sorted_reco = sorted(reco.items(), key=lambda x: x[1][sort_by] if x[1][sort_by] != 'N/A' else -float('inf'), reverse=True)
            
            st.markdown("### Recommandations")
            cols = st.columns(3)
            for i, (t, row) in enumerate(sorted_reco):
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div style='background-color:#f8f9fa;padding:15px;border-radius:8px;margin:10px 0;'>
                            <h4>{t}</h4>
                            <p><strong>Score:</strong> {row['Score']}/5</p>
                            <p><strong>P/E:</strong> {row['P/E'] if row['P/E'] != 'N/A' else 'N/A'}</p>
                            <p><strong>Croissance:</strong> {row['Croissance'] if row['Croissance'] != 'N/A' else 'N/A'}</p>
                            <button style='background-color:#1a75ff;color:white;border-radius:5px;padding:5px 10px;border:none;'>Acheter</button>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.markdown("#### Comparaison des Scores")
            df_reco = pd.DataFrame(reco).T
            fig_reco, ax = plt.subplots()
            sns.barplot(x=df_reco.index, y=df_reco['Score'], ax=ax, palette="viridis")
            ax.set_ylabel("Score de Recommandation")
            st.pyplot(fig_reco)
            
        else:
            st.write("Aucune donnée disponible pour les recommandations.")

# --- Page: Gestion Avancée des Risques ---
elif page == "Gestion Avancée des Risques":
    st.title("⚡ Gestion Avancée des Risques")
    
    tickers_input = st.text_input("Symboles ou compagnies (séparés par des virgules)", key="risk", placeholder="Ex: AAPL,MSFT,GOOGL")
    
    with st.expander("⚙️ Paramètres de risque", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Période des données", ["1mo", "3mo", "6mo", "1y"], index=3)
        with col2:
            confidence_level = st.slider("Niveau de confiance VaR/CVaR (%)", 90, 99, 95)
    
    if tickers_input:
        tl = [t.strip().upper() for t in tickers_input.split(',')]
        if not tl:
            st.warning("Veuillez saisir au moins un ticker valide.")
        else:
            invalid_tickers = []
            data_dict = {}
            for t in tl:
                data = get_history(t, period=period)
                if data is None:
                    invalid_tickers.append(t)
                else:
                    data_dict[t] = data['Close']
            
            if invalid_tickers:
                st.error(f"Impossible de récupérer les données pour les tickers suivants : {', '.join(invalid_tickers)}. Essayez d'autres tickers (par exemple, AAPL, MSFT).")
            else:
                data = pd.DataFrame(data_dict)
                
                if data.empty:
                    st.error(f"Aucune donnée disponible pour les tickers {', '.join(tl)} sur la période {period}. Vérifiez les tickers ou essayez une autre période.")
                else:
                    rets = data.pct_change().dropna()
                    if rets.empty or len(rets) < 2:
                        st.error(f"Données insuffisantes pour calculer les rendements. Assurez-vous que les tickers {', '.join(tl)} ont suffisamment de données sur la période {period}.")
                    else:
                        alpha = (100 - confidence_level) / 100
                        var1 = rets.quantile(alpha)
                        var5 = var1 * np.sqrt(5)
                        var10 = var1 * np.sqrt(10)
                        cvar1 = rets[rets.le(var1)].mean()
                        vol = rets.std() * np.sqrt(252) * 100

                        st.markdown("### Analyse des Risques")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### VaR et CVaR")
                            var_df = pd.DataFrame({
                                '1 Jour': var1 * 100,
                                '5 Jours': var5 * 100,
                                '10 Jours': var10 * 100
                            }, index=tl)
                            cvar_df = pd.DataFrame({
                                '1 Jour': cvar1 * 100,
                                '5 Jours': cvar1 * np.sqrt(5) * 100,
                                '10 Jours': cvar1 * np.sqrt(10) * 100
                            }, index=tl)
                            st.table(var_df.style.format("{:.2f}%").set_caption("VaR (%)"))
                            st.table(cvar_df.style.format("{:.2f}%").set_caption("CVaR (%)"))
                        
                        with col2:
                            st.markdown("#### Volatilité Annualisée")
                            fig_gauge, ax = plt.subplots()
                            sns.barplot(x=vol, y=tl, ax=ax, palette="Blues_d")
                            ax.set_xlabel("Volatilité (%)")
                            st.pyplot(fig_gauge)
                        
                        st.markdown("#### Simulateur de Scénarios")
                        scenario = st.selectbox("Choisir un scénario", ["Chute de marché (-10%)", "Hausse de marché (+10%)", "Volatilité élevée"])
                        if scenario == "Chute de marché (-10%)":
                            sim_rets = rets * 1.1
                        elif scenario == "Hausse de marché (+10%)":
                            sim_rets = rets * 0.9
                        else:
                            sim_rets = rets * np.random.normal(1, 0.02, len(rets))
                        sim_var = sim_rets.quantile(alpha) * 100
                        st.metric("VaR simulée (1 Jour)", f"{sim_var.mean():.2f}%")

# --- Page: Optimisation de Portefeuille ---
elif page == "Optimisation de Portefeuille":
    st.title("📊 Optimisation de Portefeuille")
    tickers_input = st.text_input("Symboles ou compagnies (séparés par des virgules)", key="opt", placeholder="Ex: AAPL,MSFT,GOOGL")

    with st.expander("⚙️ Paramètres d'optimisation", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Période des données", ["1y", "2y", "5y"], index=0)
        with col2:
            sims = st.slider("Nombre de simulations", 1000, 10000, 5000, step=1000)

    if tickers_input:
        tl = [t.strip().upper() for t in tickers_input.split(',')]
        if len(tl) == 0:
            st.warning("Veuillez saisir au moins un ticker valide.")
        else:
            invalid_tickers = []
            data_dict = {}
            for t in tl:
                data = get_history(t, period=period)
                if data is None:
                    invalid_tickers.append(t)
                else:
                    data_dict[t] = data['Close']
            
            if invalid_tickers:
                st.error(f"Impossible de récupérer les données pour les tickers suivants : {', '.join(invalid_tickers)}. Essayez d'autres tickers (par exemple, AAPL, MSFT).")
            else:
                data = pd.DataFrame(data_dict)
                if data.empty:
                    st.error(f"Aucune donnée disponible pour les tickers {', '.join(tl)} sur la période sélectionnée ({period}). Vérifiez les tickers ou essayez une autre période.")
                else:
                    rets = data.pct_change().dropna()
                    if rets.empty or len(rets) < 2:
                        st.error(f"Données insuffisantes pour effectuer l'optimisation. Assurez-vous que les tickers {', '.join(tl)} ont suffisamment de données sur la période {period}.")
                    else:
                        n = len(tl)
                        results = np.zeros((3, sims))
                        weights_list = []
                        for i in range(sims):
                            w = np.random.dirichlet(np.ones(n))
                            weights_list.append(w)
                            ret = np.sum(rets.mean() * w) * 252
                            vol = np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))
                            results[:, i] = [vol, ret, ret / vol]
                        idx = np.argmax(results[2])
                        w_opt = weights_list[idx]

                        st.markdown("### Résultats de l'Optimisation")
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown("#### Frontière Efficiente")
                            fig, ax = plt.subplots()
                            scatter = ax.scatter(results[0] * 100, results[1] * 100, c=results[2], cmap='viridis')
                            ax.scatter(results[0, idx] * 100, results[1, idx] * 100, c='red', s=100, label='Portefeuille Optimal')
                            ax.set_xlabel("Volatilité Annualisée (%)")
                            ax.set_ylabel("Rendement Annualisé (%)")
                            ax.legend()
                            plt.colorbar(scatter, label='Ratio Sharpe')
                            st.pyplot(fig)

                        with col2:
                            st.markdown("#### Répartition des Poids Optimaux")
                            fig_pie, ax_pie = plt.subplots()
                            ax_pie.pie(w_opt, labels=tl, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("muted"))
                            ax_pie.axis('equal')
                            st.pyplot(fig_pie)

                        st.markdown("#### Métriques Clés")
                        col_metrics = st.columns(3)
                        opt_return = np.sum(rets.mean() * w_opt) * 252 * 100
                        opt_vol = results[0, idx] * 100
                        sharpe_ratio = results[2, idx]
                        with col_metrics[0]:
                            st.metric("Rendement Annualisé", f"{opt_return:.2f}%")
                        with col_metrics[1]:
                            st.metric("Volatilité Annualisée", f"{opt_vol:.2f}%")
                        with col_metrics[2]:
                            st.metric("Ratio Sharpe", f"{sharpe_ratio:.2f}")

                        st.markdown("#### Poids Optimaux")
                        df_opt = pd.DataFrame({'Actif': tl, 'Poids Optimal': [f"{w * 100:.2f}%" for w in w_opt]})
                        st.table(df_opt.style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'border': '1px solid #ddd',
                            'padding': '8px'
                        }))

# --- Page: Dashboard Personnalisé ---
elif page == "Dashboard Personnalisé":
    st.title("🎨 Dashboard Personnalisé")
    
    tickers_input = st.text_input("Symboles ou compagnies (séparés par des virgules)", key="dash", placeholder="Ex: AAPL,MSFT,GOOGL")
    
    st.markdown("### Choisir les Widgets")
    col1, col2 = st.columns(2)
    with col1:
        show_price = st.checkbox("Prix Actuel", value=True)
        show_chart = st.checkbox("Graphique Historique", value=True)
    with col2:
        show_ratios = st.checkbox("Ratios Financiers", value=True)
        show_news = st.checkbox("Actualités Récentes", value=True)
    
    if tickers_input:
        tl = [t.strip().upper() for t in tickers_input.split(',')]
        for t in tl:
            ticker_data = get_ticker_data(t)
            if ticker_data is None:
                st.error(f"Impossible de récupérer les données pour {t}. Essayez un autre ticker (par exemple, AAPL ou MSFT).")
                continue
                
            st.markdown(f"### {t}")
            data = ticker_data.get("historical_data")
            info = ticker_data.get("info")
            news = ticker_data.get("news")
            
            if show_price:
                if data is not None and not data.empty and "Close" in data.columns:
                    price = data["Close"].iloc[-1]
                    st.metric(label=f"Prix actuel de {t}", value=f"{float(price):.2f} $")
                else:
                    st.write(f"Aucune donnée disponible pour afficher le prix actuel de {t}.")
            
            if show_chart:
                if data is not None and not data.empty and "Close" in data.columns:
                    st.line_chart(data["Close"])
                else:
                    st.write(f"Aucune donnée disponible pour afficher le graphique historique de {t}.")
            
            if show_ratios:
                ratios = get_ratios(t)
                if ratios:
                    st.subheader("Ratios Financiers")
                    st.table(pd.DataFrame.from_dict(ratios, orient='index', columns=['Valeur']))
                else:
                    st.write(f"Aucune donnée disponible pour les ratios financiers de {t}.")
            
            if show_news:
                st.subheader("Actualités Récentes")
                if news:
                    for n in news[:5]:
                        title = n.get("title", "N/A")
                        st.write(f"- {title}")
                else:
                    st.write(f"Aucune actualité récente disponible pour {t}.")

# --- Page: Suivi Temps Réel du Portefeuille ---
elif page == "Suivi Temps Réel du Portefeuille":
    st.title("⏱️ Suivi Temps Réel du Portefeuille")
    
    st.write("Saisissez jusqu'à 20 titres maximum (séparés par des virgules). Ex: AAPL,MSFT,GOOGL")
    tickers_input = st.text_input("Symboles ou compagnies", key="realtime", placeholder="Ex: AAPL,MSFT,GOOGL")
    
    if tickers_input:
        tl = [t.strip().upper() for t in tickers_input.split(',')]
        if len(tl) > 20:
            st.error("Vous avez saisi plus de 20 titres. Veuillez limiter à 20 titres maximum.")
        else:
            portfolio_data = {}
            for t in tl:
                ticker_data = get_ticker_data(t)
                if ticker_data and ticker_data.get("historical_data") is not None:
                    data = ticker_data["historical_data"]
                    if not data.empty and "Close" in data.columns:
                        price = data["Close"].iloc[-1]
                        # Calculer la variation sur le dernier jour (si disponible)
                        if len(data["Close"]) >= 2:
                            prev_price = data["Close"].iloc[-2]
                            change = price - prev_price
                            change_pct = (change / prev_price) * 100
                        else:
                            change = "N/A"
                            change_pct = "N/A"
                        portfolio_data[t] = {"price": price, "change": change, "change_pct": change_pct}
                    else:
                        st.warning(f"Aucune donnée disponible pour {t}.")
                else:
                    st.warning(f"Impossible de récupérer les données pour {t}.")
            
            if portfolio_data:
                st.subheader("Valeur Actuelle du Portefeuille")
                cols = st.columns(3)  # Afficher les données en 3 colonnes pour une meilleure lisibilité
                for i, (t, data) in enumerate(portfolio_data.items()):
                    with cols[i % 3]:
                        st.metric(
                            label=f"Prix de {t}",
                            value=f"{float(data['price']):.2f} $",
                            delta=f"{float(data['change']):.2f} $ ({float(data['change_pct']):.2f}%)" if data['change'] != "N/A" else None,
                            delta_color="normal" if data['change'] == "N/A" else ("inverse" if data['change'] < 0 else "normal")
                        )
            else:
                st.write("Aucune donnée disponible pour le suivi en temps réel.")

# --- Page: Export & Reporting ---
elif page == "Export & Reporting":
    st.title("📄 Export & Reporting")
    
    tickers_input = st.text_input("Symboles ou compagnies (séparés par des virgules)", key="export", placeholder="Ex: AAPL,MSFT,GOOGL")
    
    if tickers_input:
        tl = [t.strip().upper() for t in tickers_input.split(',')]
        report_data = []
        for t in tl:
            ticker_data = get_ticker_data(t)
            if ticker_data and ticker_data.get("historical_data") is not None:
                data = ticker_data["historical_data"]
                ratios = get_ratios(t)
                if data is not None and not data.empty and "Close" in data.columns:
                    latest_price = data["Close"].iloc[-1]
                    report_data.append({
                        "Ticker": t,
                        "Prix Actuel ($)": latest_price,
                        "P/E Ratio": ratios.get("PER", "N/A") if ratios else "N/A",
                        "Volatilité Annualisée (%)": (data["Close"].pct_change().std() * np.sqrt(252) * 100) if len(data["Close"]) > 1 else "N/A"
                    })
                else:
                    st.warning(f"Aucune donnée historique disponible pour {t}.")
            else:
                st.warning(f"Impossible de récupérer les données pour {t}.")
        
        if report_data:
            df_report = pd.DataFrame(report_data)
            st.table(df_report)
            
            csv = df_report.to_csv(index=False)
            st.download_button(
                label="Télécharger le rapport (CSV)",
                data=csv,
                file_name="rapport_investissement.csv",
                mime="text/csv"
            )
        else:
            st.write("Aucune donnée disponible pour générer un rapport.")

# --- Footer ---
st.markdown(
    """
    <div class="custom-footer">
        © 2025 Argentis Investment - Tous droits réservés
    </div>
    """,
    unsafe_allow_html=True
)