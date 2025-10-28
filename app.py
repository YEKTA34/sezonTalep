import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Sezonluk Talep Tahminleme", layout="wide")

st.markdown(
    """
    <style>
    .metric-box {background:#2c3e50;color:#fff;padding:16px;border-radius:14px;text-align:center;box-shadow:2px 2px 10px rgba(0,0,0,.25)}
    .metric-label{font-size:14px;color:#ecf0f1;margin-bottom:4px}
    .metric-value{font-size:22px;font-weight:700}
    .card{background:#ffffff;border-radius:14px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,.08)}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_uploaded_excel(file):
    df = pd.read_excel(file)
    df = df.rename(columns={
        'tarih':'Tarih','Tarih':'Tarih',
        'urun':'Ürün','Urün':'Ürün','Ürün':'Ürün','product':'Ürün',
        'satis':'Satış','Satis':'Satış','Sales':'Satış'
    })
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df['Ürün'] = df['Ürün'].astype(str)
    df['Satış'] = pd.to_numeric(df['Satış'], errors='coerce').fillna(0)
    df = (
        df.set_index('Tarih')
          .groupby('Ürün')
          .resample('M')['Satış']
          .sum()
          .reset_index()
    )
    return df

@st.cache_data
def create_dummy_data():
    np.random.seed(42)
    dates = pd.date_range('2021-01-01','2024-12-31',freq='M')
    products = ['Bot','Sneaker','Sandalet','Topuklu Ayakkabı']
    rows = []
    for p in products:
        for d in dates:
            m = d.month
            if p=='Bot':
                base = 400 if m in [12,1,2] else 110
            elif p=='Sandalet':
                base = 360 if m in [6,7,8] else 85
            elif p=='Sneaker':
                base = 210
            else:
                base = 160 if m in [3,4,5,9,10] else 95
            val = max(0, int(base + np.random.normal(0, 22)))
            rows.append([d, p, val])
    ddf = pd.DataFrame(rows, columns=['Tarih','Ürün','Satış'])
    return ddf

@st.cache_data
def seasonal_labels():
    return {
        'Kış': [12,1,2],
        'İlkbahar': [3,4,5],
        'Yaz': [6,7,8],
        'Sonbahar': [9,10,11]
    }

def simple_forecast(series: pd.Series, horizon: int = 6):
    by_month = series.groupby(series.index.month).mean()
    last12 = series.tail(12)
    base = last12.mean() if len(last12)>0 else series.mean()
    fc = []
    idx = []
    last_date = series.index.max()
    for i in range(1, horizon+1):
        next_date = (last_date + pd.offsets.MonthEnd(i))
        m = next_date.month
        seasonal = by_month.get(m, base)
        value = 0.5*base + 0.5*seasonal  # basit karışım
        fc.append(max(0, float(value)))
        idx.append(next_date)
    fdf = pd.DataFrame({'Tarih': idx, 'Tahmin': fc})
    return fdf

def prophet_forecast(df_prod: pd.DataFrame, horizon: int = 6):
    try:
        from prophet import Prophet
    except Exception:
        return None, 'Prophet kütüphanesi yüklü değil.'
    d = df_prod.rename(columns={'Tarih':'ds','Satış':'y'})[['ds','y']]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(d)
    future = m.make_future_dataframe(periods=horizon, freq='M')
    fc = m.predict(future)
    return fc[['ds','yhat','yhat_lower','yhat_upper']], None

with st.sidebar:
    st.header("📂 Veri Kaynağı")
    f = st.file_uploader("Excel Yükle (Kolonlar: Tarih, Ürün, Satış)", type=["xlsx","xls","csv"]) 
    if f is not None and f.name.lower().endswith('.csv'):
        raw = pd.read_csv(f)
        tmp = BytesIO()
        raw.to_excel(tmp, index=False)
        tmp.seek(0)
        df_data = load_uploaded_excel(tmp)
    elif f is not None:
        df_data = load_uploaded_excel(f)
    else:
        df_data = create_dummy_data()
        st.caption("Örnek veri kullanılıyor. Excel yüklersen gerçek verinle çalışır.")

    pages = ["Dashboard","Ürün Tahmini","Sezon Analizi","Raporlar","Ayarlar"]
    page = st.radio("Sayfa", pages, index=0)

all_products = sorted(df_data['Ürün'].unique())
df_data = df_data.sort_values('Tarih')

if page == "Dashboard":
    st.title("📊 Genel Dashboard")
    total_sales = int(df_data['Satış'].sum())
    months = df_data['Tarih'].dt.to_period('M').nunique()
    avg_month = int(total_sales / max(1, months))
    years = df_data['Tarih'].dt.year.nunique()

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Toplam Satış</div><div class='metric-value'>{total_sales:,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Aylık Ortalama</div><div class='metric-value'>{avg_month:,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Yıl Sayısı</div><div class='metric-value'>{years}</div></div>", unsafe_allow_html=True)

    st.subheader("Aylık Toplam Satış (Tüm Ürünler)")
    monthly = (
        df_data.assign(Ay=lambda d: d['Tarih'].dt.to_period('M').dt.to_timestamp())
               .groupby('Ay')['Satış'].sum().reset_index()
    )
    fig = px.line(monthly, x='Ay', y='Satış')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ürün x Ay Isı Haritası")
    hm = df_data.copy()
    hm['Ay'] = hm['Tarih'].dt.month
    pivot = hm.pivot_table(index='Ürün', columns='Ay', values='Satış', aggfunc='sum').fillna(0)
    fig2 = px.imshow(pivot, aspect='auto', labels=dict(color='Satış'))
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Ürün Tahmini":
    st.title("🔮 Ürün Bazlı Tahmin ve Sipariş Önerisi")
    colA, colB, colC = st.columns([2,1,1])
    with colA:
        prod = st.selectbox("Ürün Seç", all_products)
    with colB:
        horizon = st.number_input("Tahmin Ufku (Ay)", min_value=1, max_value=12, value=6)
    with colC:
        stock = st.number_input("Mevcut Stok", min_value=0, value=500)

    dfp = df_data[df_data['Ürün']==prod].copy()
    dfp = dfp.set_index('Tarih').asfreq('M').fillna(0).reset_index()

    model_opt = st.radio("Model", ["Basit (MA+Mevsim)", "Prophet"], horizontal=True)

    if model_opt == "Prophet":
        fc, err = prophet_forecast(dfp[['Tarih','Satış']], horizon)
        if err:
            st.warning(err + " — Basit tahmine geri dönüldü.")
            fc_simple = simple_forecast(dfp.set_index('Tarih')['Satış'], horizon)
            fc_plot = fc_simple.rename(columns={'Tarih':'ds','Tahmin':'yhat'})
        else:
            fc_plot = fc.rename(columns={'ds':'Tarih','yhat':'Tahmin'})
            fc_plot = fc_plot.tail(horizon)
    else:
        fc_simple = simple_forecast(dfp.set_index('Tarih')['Satış'], horizon)
        fc_plot = fc_simple

    st.subheader(f"{prod} Satış Geçmişi ve Tahmin")
    hist = dfp[['Tarih','Satış']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['Tarih'], y=hist['Satış'], mode='lines', name='Gerçek'))
    fig.add_trace(go.Scatter(x=fc_plot['Tarih'], y=fc_plot['Tahmin'], mode='lines+markers', name='Tahmin'))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)

    total_forecast = int(np.ceil(fc_plot['Tahmin'].sum()))
    need = max(0, total_forecast - int(stock))

    c1,c2,c3 = st.columns(3)
    c1.metric("Tahmini  Dönem Satışı", f"{total_forecast}")
    c2.metric("Mevcut Stok", f"{int(stock)}")
    c3.metric("Önerilen Sipariş", f"{need}")

    st.dataframe(fc_plot.rename(columns={'Tahmin':'Tahmini Satış'}), use_container_width=True)

elif page == "Sezon Analizi":
    st.title("🍂 Sezon Bazlı Talep Analizi")
    seasons = seasonal_labels()
    left, right = st.columns([1,3])
    with left:
        season = st.selectbox("Sezon Seç", list(seasons.keys()), index=2)
        months = seasons[season]
    sdf = df_data.copy()
    sdf['Ay'] = sdf['Tarih'].dt.month
    sdf['Yıl'] = sdf['Tarih'].dt.year
    season_df = sdf[sdf['Ay'].isin(months)]

    st.subheader(f"{season} Dönemi Ürün Sıralaması (Toplam Satış)")
    rank = season_df.groupby('Ürün')['Satış'].sum().sort_values(ascending=False).reset_index()
    figr = px.bar(rank, x='Ürün', y='Satış')
    st.plotly_chart(figr, use_container_width=True)

    st.subheader(f"{season} Dönemi Yıla Göre Karşılaştırma")
    comp = season_df.groupby(['Yıl','Ürün'])['Satış'].sum().reset_index()
    figc = px.line(comp, x='Yıl', y='Satış', color='Ürün', markers=True)
    st.plotly_chart(figc, use_container_width=True)

elif page == "Raporlar":
    st.title("📄 Raporlar ve Dışa Aktarım")
    st.caption("Seçilen sezona göre bir sonraki dönem için sipariş öneri raporu oluştur.")
    seasons = seasonal_labels()
    season = st.selectbox("Sezon", list(seasons.keys()), index=2)
    months = seasons[season]
    horizon = len(months)

    rep_rows = []
    for p in all_products:
        dfp = df_data[df_data['Ürün']==p].copy()
        dfp = dfp.set_index('Tarih').asfreq('M').fillna(0)
        fc = simple_forecast(dfp['Satış'], horizon)
        total_fc = int(np.ceil(fc['Tahmin'].sum()))
        rep_rows.append([p, total_fc])
    rep = pd.DataFrame(rep_rows, columns=['Ürün','Tahmini Talep'])

    st.subheader("🧾 Ürün Bazlı Tahmini Talep (Sezon)")
    st.dataframe(rep, use_container_width=True)

    st.markdown("**Stok Girişi (Opsiyonel):** Ürün başına mevcut stokunuzu girin, sipariş önerisi hesaplayalım.")
    defaults = {p: 500 for p in all_products}
    with st.form("stok_form"):
        cols = st.columns(4)
        stocks = {}
        for i,p in enumerate(all_products):
            with cols[i%4]:
                stocks[p] = st.number_input(f"{p} stok", min_value=0, value=defaults[p])
        submitted = st.form_submit_button("Sipariş Önerisini Hesapla")

    if 'stocks' in locals() and (submitted or True):
        rep['Mevcut Stok'] = rep['Ürün'].map(stocks) if 'stocks' in locals() else 0
        rep['Önerilen Sipariş'] = (rep['Tahmini Talep'] - rep['Mevcut Stok']).clip(lower=0)
        st.subheader("📦 Sipariş Öneri Tablosu")
        st.dataframe(rep, use_container_width=True)

        try:
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
                rep.to_excel(writer, index=False, sheet_name='SiparisOneri')
            xbytes = bio.getvalue()
            st.download_button("Excel İndir", data=xbytes, file_name="siparis_oneri.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.info("Excel oluşturulamadı, CSV indirabilirsiniz.")
        st.download_button("CSV İndir", data=rep.to_csv(index=False), file_name="siparis_oneri.csv", mime="text/csv")

elif page == "Ayarlar":
    st.title("⚙️ Ayarlar & Yardım")
    st.markdown("""
    **Veri formatı**: Excel/CSV'de `Tarih`, `Ürün`, `Satış` kolonları olmalı. Tarihler aylık veya gün/haftalık olabilir; uygulama aylığa toplulaştırır.
    
    **Modeller**:
    - *Basit (MA+Mevsim)*: Kütüphane kurulum gerektirmez, hızlı ve hafif.
    - *Prophet*: Daha gerçekçi mevsimsellik ve trend yakalar. Sunucunuzda `prophet` kurulu olmalı.
    
    **İpuçları**:
    - Yüksek sezonda stok açığı riskini görmek için Raporlar sayfasında stokları ürün bazında girin.
    - Veri aralığı 2+ yıl ise Prophet daha iyi sonuç verir.
    """)
