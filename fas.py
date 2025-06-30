import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Modatna", layout="wide")

#— Raw GitHub URL for the CSV
url = "https://github.com/ruth1445/modatna/raw/main/Womens%20Clothing%20E-Commerce%20Reviews.csv"
st.write(f"CSV File URL: {url}")

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

lottie_header = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_yr6zz3wv.json")
lottie_cherry = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_s2lhbzqf.json")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Serif+4:wght@400;600&display=swap');

/* Soft gradient + grain */
.stApp {
  background:
    linear-gradient(180deg, #FDF6F0 0%, #FFF7E5 100%),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.7' numOctaves='3'/></filter><rect width='100%' height='100%' fill='%23fff' filter='url(%23n)'/></svg>");
  background-blend-mode: overlay;
  background-size: cover, 150px 150px;
  background-repeat: no-repeat, repeat;
  position: relative;
}

/* Cherry blossom petals overlay */
body::before {
  content: "";
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: url('https://i.imgur.com/9PWj3Po.png') center/cover no-repeat;
  opacity: 0.04;
  pointer-events: none;
  z-index: -1;
}

/* Transparent Lottie Containers */
.lottie-container, .streamlit-lottie, canvas, div[class*="lottie"] {
  background: transparent !important;
}

/* Slide-in Animation */
@keyframes fadeSlide {
  from { transform: translateY(20px); opacity: 0; }
  to   { transform: translateY(0); opacity: 1; }
}
.block-container, .stContainer > div {
  animation: fadeSlide 0.8s ease-out forwards;
}

/* Headers */
h1, h2 {
  font-family: 'Playfair Display', serif !important;
  color: #493C3C !important;
  text-align: center;
  margin: 0;
}
h2 {
  font-size: 1.5rem;
  font-weight: 500;
  margin-bottom: 10px;
}

/* Body Text */
.css-18e3th9 p, .css-18e3th9 {
  font-family: 'Source Serif 4', serif;
  color: #3A2C27;
  font-size: 1rem;
}

/* Inputs & Tabs */
[data-testid="stTextInput"] input,
[data-testid="stMultiselect"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] > div > div {
  background: rgba(255,255,255,0.6) !important;
  backdrop-filter: blur(3px);
  border-radius: 8px;
  color: #3A2C27;
}
.stTabs [role="tab"] {
  font-family: 'Playfair Display', serif;
  font-size: 1rem;
  color: #7a5c56;
}
.stTabs [role="tab"][data-selected='true'] {
  color: #493C3C;
  border-bottom: 2px solid #493C3C;
}

/* Transparent Plotly */
.js-plotly-plot .plot-container .svg-container {
  background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

if lottie_header:
    st_lottie(lottie_header, height=180, key="header", quality="high")
if lottie_cherry:
    st_lottie(lottie_cherry, height=200, key="petals", loop=True, quality="high")

@st.cache_data
def load_and_process_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=['Class Name','Rating','Title'])
    price_map = {'Blouses':45, 'Dresses':80, 'Pants':60, 'Jackets':120, 'Sweaters':70}
    df['Original Price']     = df['Class Name'].map(price_map).fillna(50)
    df['Value Retention %']  = df['Rating'] / 5
    df['Resale Price']       = df['Original Price'] * df['Value Retention %']
    texts = df['Title'].astype(str)
    X = TfidfVectorizer(stop_words='english', max_features=500).fit_transform(texts)
    df['Cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(X)
    coords = PCA(n_components=2).fit_transform(X.toarray())
    df['PCA1'], df['PCA2'] = coords[:,0], coords[:,1]
    style_map = {
        0: 'Cottagecore Vintage',
        1: 'Minimalist Luxe',
        2: 'Playful Femme',
        3: 'Classic Formal',
        4: 'Street Chic'
    }
    df['Style Label'] = df['Cluster'].map(style_map)
    return df

df = load_and_process_data(url)

st.markdown("<h1>Modatna</h1><h2>Insights</h2>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💰 Value Trends", "🌸 Style Archetypes"])

with tab1:
    st.header("Categories That Hold Their Value")
    cats     = sorted(df['Class Name'].unique())
    selected = st.multiselect("Filter Categories", options=cats, default=cats)
    avg      = (
        df[df['Class Name'].isin(selected)]
        .groupby('Class Name')['Resale Price']
        .mean()
        .reset_index()
    )
    min_price = st.slider(
        "Min Avg Resale ($)",
        float(avg['Resale Price'].min()),
        float(avg['Resale Price'].max()),
        float(avg['Resale Price'].quantile(0.25))
    )
    filt = avg[avg['Resale Price'] >= min_price]
    fig1 = px.bar(
        filt,
        x='Class Name', y='Resale Price',
        labels={'Resale Price':'Avg Resale ($)'},
        color='Resale Price',
        color_continuous_scale='Purples'
    )
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Explore Style Archetypes")
    min_r = st.slider("Min Value Retention %", 0.0, 1.0, 0.2, step=0.05)
    df2   = df[df['Value Retention %'] >= min_r]
    kw    = st.text_input("Search Titles")
    if kw:
        df2 = df2[df2['Title'].str.contains(kw, case=False, na=False)]

    fig2 = px.scatter(
        df2, x='PCA1', y='PCA2', color='Style Label',
        hover_data=['Title','Resale Price'],
        title="Style Cluster Projection"
    )
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    sel = st.selectbox("Highlight Style", sorted(df2['Style Label'].unique()))
    sub = df2[df2['Style Label'] == sel]
    fig3 = px.scatter(
        sub, x='PCA1', y='PCA2', color='Style Label',
        hover_data=['Title','Resale Price'],
        title=f"Items in {sel}"
    )
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ by Ruth Sharon</p>",
    unsafe_allow_html=True
)
