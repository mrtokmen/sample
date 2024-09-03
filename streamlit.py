import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
st.set_page_config(layout="wide")
@st.cache_data
def get_data():
    df = pd.read_csv("datasets/deprem-senaryosu-analiz-sonuclar.csv",encoding="ISO-8859-9", delimiter=";")
    return df

# def get_model():
#    model = joblib.load("ibb_model.joblib")
#    return model

st.header("🏙 :red[İBB] DEPREM SENARYOSU :red[ANALİZİ] 🏙")
tab_home,tab_graphics,tab_model, tab_comment= st.tabs(["Ana Sayfa","Grafikler","Model","Model Yorumu "])

sol_taraf,sag_taraf = tab_home.columns(2,gap="large")
sol_taraf.subheader(" - :red[İstanbul'un Deprem Gerçeği ve Olası Büyük Etkileri ] :" ,divider="orange")
sol_taraf.markdown(" * İstanbul, tarih boyunca büyük depremlerle sarsılmış bir şehir olarak, gelecekte yaşanması muhtemel yeni sarsıntılara karşı da yüksek risk taşıyor. Bu kadim şehirde olası bir deprem senaryosu, milyonlarca insanın hayatını, kentsel altyapıyı ve ekonomik dengeleri derinden etkileyecek güçte olabilir. "
                   "Bu nedenle, İstanbul’un depreme hazırlıklı olması, hayati bir öneme sahiptir ve bu hazırlığın temel taşları, doğru risk analizleri ve etkili müdahale stratejilerinden geçmektedir.")
sol_taraf.image("media/deprem.webp",width=700, caption='😥Olası Bir İstanbul Depremi😥')#use_column_width=True)
sag_taraf.subheader(" - :red[Veri Seti Hakkında] :",divider="orange")
sag_taraf.markdown(" * Bu veri seti, İstanbul için 7.5 Mw büyüklüğünde ve gece gerçekleşmesi öngörülen bir deprem senaryosuna dayanan analiz sonuçlarını içeriyor. Bu analizler, "
            "şehrin farklı bölgelerinde olası hasar senaryolarını, acil durum müdahale planlarını ve etkilenme derecelerini değerlendirmek için yapılmıştır. "
            "Senaryo kapsamında, olası can kayıpları, yaralanmalar, bina hasarları ve altyapı üzerindeki etkiler gibi faktörler de dikkate alınmıştır."
            " Bu analizler, hem risklerin minimize edilmesine yönelik stratejilerin geliştirilmesi hem de gelecekteki afet yönetimi planlarının optimize "
            "edilmesi açısından büyük önem taşımaktadır. Veri seti, çeşitli parametreler ve tahmin modelleri kullanılarak elde edilen detaylı sonuçları içerir ve şehir planlamacılarının,"
            " acil durum müdahale ekiplerinin ve ilgili diğer paydaşların karar alma süreçlerini desteklemeyi amaçlamaktadır.")

df = get_data()
sag_taraf.dataframe(df,width=900)


#########################################################################################################################

tab_graphics.subheader(":red[İlçelere göre Can Kaybı ve Ağır Yaralı Sayısı :]")

fig = px.bar(df, x='ilce_adi', y=['can_kaybi_sayisi','agir_yarali_sayisi'],
             labels={'ilce_adi': 'İlçe adı', 'can_kaybi_sayisi': 'Can kaybı sayısı',"agir_yarali_sayisi":"agir yarali sayisi"},
             color_discrete_sequence=['#ffaa00', '#00aaff'])
fig.for_each_trace(lambda t: t.update(name=t.name.replace('can_kaybi_sayisi', 'Can kaybı sayısı').replace('agir_yarali_sayisi', 'Ağır yaralı sayısı')))
fig.update_layout(
    barmode='group',
    xaxis_tickangle=-45)
# Grafiği Streamlit uygulamasında gösterme
tab_graphics.plotly_chart(fig, use_container_width=True)
#########################################################################################################################

tab_graphics.subheader(":red[İlçelere göre Çok Ağır ve Ağır Hasarlı Bina Sayısı :]")
fig = px.bar(df,
             x='ilce_adi',
             y=['cok_agir_hasarli_bina_sayisi', 'agir_hasarli_bina_sayisi'],
             labels={'value': 'Bina Sayısı', 'variable': 'Hasar Durumu',"ilce_adi":"İlçe Adı"})

# Legend etiketlerini özelleştirme
fig.update_layout(
    barmode='group',
    xaxis_tickangle=-45,
    legend_title_text='Hasar Durumu',
    legend=dict(
        traceorder='normal',
        orientation='h',
        xanchor='right',
        yanchor='top',
        x=1,
        y=1
    ),
    width=1200,  # Genişlik (px cinsinden)
    height=800  # Yükseklik (px cinsinden)
)

fig.for_each_trace(lambda t: t.update(name={
    'cok_agir_hasarli_bina_sayisi': 'Çok Ağır Hasarlı Bina Sayısı',
    'agir_hasarli_bina_sayisi': 'Ağır Hasarlı Bina Sayısı'
}.get(t.name, t.name)))

tab_graphics.plotly_chart(fig)
#######################################################################################################################
tab_graphics.subheader(":red[İlçelere göre Boru Hattı Hasarları :]")
selected_ilceler = tab_graphics.multiselect(
    'Seçmek istediğiniz ilçeleri seçin:',
    options=df['ilce_adi'].unique(),
    placeholder='Bir veya daha fazla ilçe seçin')

if selected_ilceler:
    filtered_df = df[df['ilce_adi'].isin(selected_ilceler)]
    totals = {
        'Doğalgaz Boru Hasarı': filtered_df['dogalgaz_boru_hasari'].sum(),
        'İçme Suyu Boru Hasarı': filtered_df['icme_suyu_boru_hasari'].sum(),
        'Atık Su Boru Hasarı': filtered_df['atik_su_boru_hasari'].sum()
    }

    with tab_graphics:
        for key, value in totals.items():
            st.write(f'Seçilen ilçelerdeki toplam {key.lower()}: {value}')


        summary_df = pd.DataFrame(list(totals.items()), columns=['Hasar Türü', 'Toplam Hasar'])
        fig = px.bar(
            summary_df,
            x='Hasar Türü',
            y='Toplam Hasar',
            title='Seçilen İlçelerdeki Hasar Türlerinin Toplamları',
            labels={'Toplam Hasar': 'Toplam Hasar Miktarı'},
            color='Hasar Türü',
            color_discrete_sequence=['#ffaa00', '#00aaff', '#ff5500']
        )
        fig.update_layout(width=400, height=400)
        tab_graphics.plotly_chart(fig, use_container_width=True)
#######################################################################################################################
# Modeli yükleme fonksiyonu
def get_model():
    model = joblib.load("ibb_model.joblib")
    return model

# Uygulama başlığı
tab_model.title(':red[Geçici Barınma Tahmin Uygulaması : ]')

# Kullanıcıdan ilçeleri seçmesini isteyin
selected_ilceler = tab_model.multiselect(
    'Seçmek istediğiniz ilçeleri seçin:',
    options=df['ilce_adi'].unique()
)

# Seçilen ilçelere göre mahalleleri filtreleyin
if selected_ilceler:
    filtered_mahalleler = df[df['ilce_adi'].isin(selected_ilceler)]['mahalle_adi'].unique()
else:
    filtered_mahalleler = []

# Kullanıcıdan mahalleleri seçmesini isteyin
selected_mahalleler = tab_model.multiselect(
    'Seçmek istediğiniz mahalleleri seçin:',
    options=filtered_mahalleler,
    placeholder='Bir veya daha fazla mahalle seçin'
)

# Kullanıcıdan çok ağır hasarlı bina sayısını al
cok_agir_hasarli_bina_sayisi = tab_model.number_input('Çok Ağır Hasarlı Bina Sayısı :', min_value=0)

# Modeli yükle
model = get_model()

# Tahmin butonu
if tab_model.button('Tahmin Et'):
    # Girdi verilerini DataFrame'e dönüştür
    input_data = pd.DataFrame({
        'cok agir hasarli bina sayisi': [cok_agir_hasarli_bina_sayisi]
    })

    # Tahmini yap
    prediction = model.predict(input_data)

    # MSE hesapla (önceden hesaplanmış mse değeri yerine dinamik olarak da hesaplanabilir)
    mse = mean_squared_error(input_data['cok agir hasarli bina sayisi'], prediction)

    # Sonuçları göster
    tab_model.write(f'Geçici Barınma Tahmini: {prediction[0]:.2f}')
    tab_model.write(f"Mean Squared Error: {mse:.2f}")
    tab_model.balloons()
########################################################################################################################
tab_comment.header(":red[Modelin Yorumlanması :]")
tab_comment.markdown(" * Bu sonuçlar, bir modelin deprem sonrası geçici barınma ihtiyacını tahmin etmek amacıyla kullanıldığını gösteriyor. Verilen değerler şu şekilde yorumlanabilir:")
tab_comment.markdown(" * Çok Ağır Hasarlı Bina Sayısı : Model,örneğin 54 tane çok ağır hasarlı bina bulunan bir senaryo üzerinde çalışıyor. Bu sayı, depremde ciddi şekilde zarar gören binaların sayısını ifade ediyor.")
tab_comment.markdown(" * Geçici Barınma Tahmini : Model, örneğin  54 çok ağır hasarlı bina olması durumunda, yaklaşık olarak 2436 kişinin geçici barınma ihtiyacında olacağını tahmin ediyor. Bu değer, modelin tahmin ettiği geçici barınma ihtiyacını yansıtır. ")
tab_comment.markdown(" * Mean Squared Error (MSE) : Bu değer, modelin tahminlerinin ne kadar hatalı olduğunu gösteren bir metrik. MSE, tahmin edilen değerler ile gerçek değerler arasındaki farkların karelerinin ortalamasıdır. Bu durumda, MSE oldukça yüksek bir değer olduğundan, modelin tahminlerinde ciddi hatalar olabileceğini gösteriyor. Yani, modelin tahminleri ile gerçek değerler arasında büyük bir fark olabilir.")
tab_comment.markdown(" :red[############################################################################################################################################################################################################################]")
tab_comment.markdown(" * Bu senaryo, hiç çok ağır hasarlı bina olmaması durumunda modelin tahminini ve performansını gösteriyor:")
tab_comment.markdown(" * Çok Ağır Hasarlı Bina Sayısı 0: Bu senaryoda, depremde çok ağır hasar görmüş hiçbir bina bulunmuyor. Bu, en düşük hasar seviyesini temsil ediyor.")
tab_comment.markdown(" * Geçici Barınma Tahmini 289.34: Model, çok ağır hasarlı bina olmamasına rağmen, yaklaşık 289 kişinin geçici barınma ihtiyacı olacağını tahmin ediyor. Bu durum, belki de başka faktörlerin (hafif hasarlı binalar, altyapı hasarı vb.) geçici barınma ihtiyacını etkilediğini gösteriyor olabilir. ")
tab_comment.markdown(" * Mean Squared Error (MSE) 83716.96: Bu MSE değeri, yine oldukça düşük. Bu, modelin bu tahmin için makul düzeyde doğru sonuçlar üretebildiğini gösteriyor. Düşük MSE değeri, modelin tahminleri ile gerçek değerler arasındaki farkın nispeten küçük olduğunu, dolayısıyla modelin bu durumda iyi performans gösterdiğini ifade eder.")