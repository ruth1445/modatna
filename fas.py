import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Modatna", layout="wide")

# Raw GitHub URL 
url = "https://raw.githubusercontent.com/ruth1445/modatna/main/Womens%20Clothing%20E-Commerce%20Reviews.csv"

st.markdown(
    f"<p style='color:white; font-size:14px;'>CSV File URL: "
    f"<a href='{url}' style='color:white; text-decoration:underline;' target='_blank'>{url}</a></p>",
    unsafe_allow_html=True
)

def load_lottieurl(u: str):
    try:
        r = requests.get(u)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

lottie_header = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_yr6zz3wv.json")
lottie_cherry = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_s2lhbzqf.json")

st.markdown("""<style>/* your CSS here */</style>""", unsafe_allow_html=True)

if lottie_header:
    st_lottie(lottie_header, height=180, key="header", quality="high")
if lottie_cherry:
    st_lottie(lottie_cherry, height=200, key="petals", loop=True, quality="high")

@st.cache_data
def load_and_process_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=['Class Name','Rating','Title'])
    price_map = {'Blouses':45,'Dresses':80,'Pants':60,'Jackets':120,'Sweaters':70}
    df['Original Price']    = df['Class Name'].map(price_map).fillna(50)
    df['Value Retention %'] = df['Rating'] / 5
    df['Resale Price']      = df['Original Price'] * df['Value Retention %']
    texts = df['Title'].astype(str)
    X = TfidfVectorizer(stop_words='english', max_features=500).fit_transform(texts)
    df['Cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(X)
    coords = PCA(n_components=2).fit_transform(X.toarray())
    df['PCA1'], df['PCA2'] = coords[:,0], coords[:,1]
    style_map = {
        0:'Cottagecore Vintage',1:'Minimalist Luxe',
        2:'Playful Femme',3:'Classic Formal',4:'Street Chic'
    }
    df['Style Label'] = df['Cluster'].map(style_map)
    return df

df = load_and_process_data(url)

st.markdown("<h1>Modatna</h1><h2>Insights</h2>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💰 Value Trends", "🌸 Style Archetypes"])

with tab1:
    st.header("Categories That Hold Their Value")
    cats = sorted(df['Class Name'].unique())
    selected = st.multiselect("Filter Categories", options=cats, default=cats)
    avg = (
        df[df['Class Name'].isin(selected)]
        .groupby('Class Name')['Resale Price']
        .mean().reset_index()
    )
    min_price = st.slider(
        "Min Avg Resale ($)",
        float(avg['Resale Price'].min()),
        float(avg['Resale Price'].max()),
        float(avg['Resale Price'].quantile(0.25))
    )
    filt = avg[avg['Resale Price'] >= min_price]
    fig1 = px.bar(
        filt, x='Class Name', y='Resale Price',
        labels={'Resale Price':'Avg Resale ($)'},
        color='Resale Price', color_continuous_scale='Purples'
    )
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Explore Style Archetypes")
    min_r = st.slider("Min Value Retention %", 0.0, 1.0, 0.2, step=0.05)
    df2 = df[df['Value Retention %'] >= min_r]
    kw = st.text_input("Search Titles")
    if kw:
        df2 = df2[df2['Title'].str.contains(kw, case=False, na=False)]

    fig2 = px.scatter(
        df2, x='PCA1', y='PCA2', color='Style Label',
        hover_data=['Title','Resale Price'], title="Style Cluster Projection"
    )
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    sel = st.selectbox("Highlight Style", sorted(df2['Style Label'].unique()))
    sub = df2[df2['Style Label'] == sel]
    fig3 = px.scatter(
        sub, x='PCA1', y='PCA2', color='Style Label',
        hover_data=['Title','Resale Price'], title=f"Items in {sel}"
    )
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ by Ruth Sharon</p>", unsafe_allow_html=True)
