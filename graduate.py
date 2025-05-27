import streamlit as st
import pandas as pd
import joblib
import pickle

st.set_page_config(layout="wide")
@st.cache_data

def get_data():
    df = pd.read_csv("bank_customer/bank_customer _churn.csv")
    return df

st.header("🏦🚶:red[BANK CUSTOMER CHURN]🏦🚶")

tab_home, tab_charts, tab_model, tab_comment= st.tabs(["Home", "Charts", "Model","Model Comment"])

# tab home ############################################################################################
column_introduction, column_dataset =tab_home.columns(2)

column_introduction.subheader(" - :red[Predicting and Preventing Customer Churn in Banking :]",divider="gray")


column_introduction.markdown("""
* This dataset has been created to analyze customer behavior and predict customer churn in a bank. It contains financial and demographic information for **10,000** customers. Each customer has a unique identification number. The dataset includes key financial indicators such as credit score, age, gender, country (France, Germany, Spain, etc.), tenure with the bank, and current account balance. Additionally, it provides details on the number of banking products owned by the customer, whether they have a credit card, and whether they are an active bank member. The estimated salary of each customer is also included.  

* One of the most important columns in the dataset is the **<span style="color:red">"churn"</span>** variable, which indicates whether a customer has left the bank. A value of **1** represents that the customer has churned, while a value of **0** means the customer has remained with the bank.  

* This dataset can be particularly useful for machine learning models to predict customer churn, improve customer relationship management, and help the bank make more informed strategic decisions. Understanding the factors that lead to customer churn can provide valuable insights for developing strategies to increase customer retention and loyalty.
""", unsafe_allow_html=True)


column_introduction.image("bank_customer/resim1.PNG", width=800)

column_dataset.subheader(" - :red[About The Dataset :]",divider="gray")

column_dataset.markdown("""
* This dataset contains various information about a bank's customers and can be used to analyze whether customers have churned (left the bank). 
The dataset includes a total of **10,000** customers, with **12** different features for each customer, such as credit score, country, gender, age, tenure (years with the bank), account balance, number of products owned, credit card ownership, active membership status, and estimated salary. Among the customers, **20.37%** have churned. 
The average credit score is **650.5**, the average age is **38.9** years, the average tenure is **5** years, the average account balance is **$76,485.89**, the average number of products owned is **1.53**, **70.55%** of customers have a credit card, and **51.51%** are active members. 
This dataset provides valuable information for analyzing customer churn and developing strategies to improve customer retention for the bank.
""")
df = get_data()
column_dataset.dataframe(df)
#######################################################################################################################
column_1, column_2 = tab_charts.columns(2)


# tab charts #####################################################################################
column_1.subheader(" - :red[Churn Status and Credit Score Analysis:]",divider="gray")
selected_country = column_1.multiselect(label='Select Country', options=df.country.unique())

if selected_country:
    filtered_df = df[df['country'].isin(selected_country)]
    selected_credit_score = column_1.multiselect(label='Determine Credit Score', options=filtered_df['credit_score'].unique())

    if selected_credit_score:
        filtered_df = filtered_df[filtered_df['credit_score'].isin(selected_credit_score)]
        col1, col2 = column_1.columns(2)
        with col1:
            column_1.write(filtered_df[['country', 'credit_score', 'gender', 'churn']])
        with col2:
            for index, row in filtered_df.iterrows():
                churn_status = 'Churn' if row['churn'] == 1 else 'Not Churn'
                column_1.write(f"Country: {row['country']}, Credit Score: {row['credit_score']}, Gender: {row['gender']}, Churn Status: {churn_status}")

#######################################################################################################
import plotly.express as px
import seaborn as sns
column_1.subheader(" - :red[Age Distribution:]",divider="gray")
sns.set(style="whitegrid")
fig = sns.histplot(df['age'], bins=30, kde=True,color='orange')
fig.set_xlabel('Age' ,fontsize=14, color='white')
fig.set_ylabel('Frequency',fontsize=14, color='white')
column_1.plotly_chart(fig.get_figure())
#######################################################################################################
import altair as alt
column_2.subheader(" - :red[Credit Score vs. Balance:]",divider="gray")
altair_chart = alt.Chart(df).mark_circle(size=10, opacity=0.7).encode(
    x=alt.X('credit_score', scale=alt.Scale(domain=(df['credit_score'].min()-10, df['credit_score'].max()+10)), title='Credit Score'),
    y=alt.Y('balance', scale=alt.Scale(domain=(df['balance'].min()-5000, df['balance'].max()+5000)), title='Balance'),
    color=alt.Color('churn:N', scale=alt.Scale(range=['orange', 'red'])),
    tooltip=['credit_score', 'balance', 'churn']
).properties(
    width=600,
    height=400
).interactive()
column_2.altair_chart(altair_chart, use_container_width=True)


#################################################################################################################################
#import altair as alt

#column_2.subheader(" - :red[Correlation Heatmap:]",divider="gray")
#numeric_df = df.select_dtypes(include=['number'])
#correlation_matrix = numeric_df.corr()
#corr_long = correlation_matrix.reset_index().melt(id_vars='index')
#corr_long.columns = ['Feature 1', 'Feature 2', 'Correlation']
#corr_long['Correlation'] = corr_long['Correlation'].apply(lambda x: f'{x:.2f}')
#heatmap = alt.Chart(corr_long).mark_rect().encode(
    #x='Feature 1:N',
    #y='Feature 2:N',
    #color='Correlation:Q',
    #tooltip=['Feature 1', 'Feature 2', 'Correlation']
#).properties(
   # width=800,
    #height=600,
#) + alt.Chart(corr_long).mark_text(baseline='middle', fontWeight='bold').encode(
    #x='Feature 1:N',
    #y='Feature 2:N',
    #text='Correlation:N')
#column_2.altair_chart(heatmap)
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Streamlit başlık
column_1.subheader(" - 📊:red[Feature Importance Visualization:📊]", divider="gray")

# Veri setini yükle
df = pd.read_csv("bank_customer/bank_customer _churn.csv")  # Veri setini buraya eklemelisin

df.drop(columns=['customer_id'], inplace=True)
# Hedef değişkeni belirle
y = df["churn"]
X = df.drop(["churn"], axis=1)

# Kategorik değişkenleri sayısal hale getirme (Label Encoding)
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # Daha sonra tersine çevirmek için kaydet

# Modelleri tanımla
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting (GBM)": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "CART (Decision Tree)": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42)
}

# Görselleştirme için boş liste
figures = []

# Modelleri eğit ve feature importance'ı al
for model_name, model in models.items():
    model.fit(X, y)

    # Feature importance'ı al
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = model.booster_.feature_importance(importance_type="gain")

    # DataFrame oluştur
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plotly Express ile görselleştirme
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        text="Importance",
        orientation="h",
        title=f"Feature Importance - {model_name}",
        color="Importance",
        color_continuous_scale="viridis"
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature", yaxis=dict(categoryorder="total ascending"))

    figures.append(fig)

# Streamlit'te grafikleri sıralı şekilde gösterme
for i in range(0, len(figures), 2):
    column_1, column_2 = tab_charts.columns(2)

    # İlk grafik
    column_1.plotly_chart(figures[i], use_container_width=True)

    # Eğer ikinci grafik varsa, onu da ekleyelim
    if i + 1 < len(figures):
        column_2.plotly_chart(figures[i + 1], use_container_width=True)


########################################################################################################################
# tab_model:
########################################################################################################################
def get_model():
    model = joblib.load("evaluate_models.joblib")
    return model

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df = pd.get_dummies(df, columns=['country'], drop_first=True)
df[['country_Germany', 'country_Spain']] = df[['country_Germany', 'country_Spain']].astype(int)
y = df["churn"]
X = df.drop(["churn"], axis=1)
classifiers = {
    "CART": (DecisionTreeClassifier(),
             {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]}),
    "RF": (RandomForestClassifier(),
           {"n_estimators": [100, 300], "max_depth": [None, 10], "min_samples_split": [2, 5]}),
    "Adaboost": (AdaBoostClassifier(algorithm="SAMME"),
                 {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}),
    "GBM": (GradientBoostingClassifier(),
            {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss", verbosity=0),
                {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}),
    "LightGBM": (LGBMClassifier(verbose=-1),
                 {"n_estimators": [100, 300], "learning_rate": [0.01, 0.1], "max_depth": [-1, 5]}),
}
def evaluate_models(X, y, cv=5):
    results = []
    for name, (model, params) in classifiers.items():
        grid_search = GridSearchCV(model, params, cv=cv, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        # Çapraz doğrulama metrikleri
        scores = cross_validate(best_model, X, y, cv=cv,
                                scoring={"accuracy": make_scorer(accuracy_score),
                                         "f1": make_scorer(f1_score),
                                         "roc_auc": make_scorer(roc_auc_score)},
                                n_jobs=-1)
        results.append({
            "Model": name,
            "Accuracy": round(np.mean(scores["test_accuracy"]), 4),
            "F1-Score": round(np.mean(scores["test_f1"]), 4),
            "ROC AUC": round(np.mean(scores["test_roc_auc"]), 4),
        })

    return results

results = evaluate_models(X, y)
results_df = pd.DataFrame(results)
tab_model = tab_model.expander(":red[Model Evaluation Results]")
tab_model.balloons()
tab_model.title(":red[Model Evaluation Results]")
tab_model.subheader(" - :red[Evaluation Metrics :]",divider="gray")
tab_model.table(results_df)
tab_model.subheader(" - :red[Model Comparison : ]",divider="gray")
accuracy_data = pd.DataFrame({
    'Model': results_df['Model'],
    'Accuracy': results_df['Accuracy']
})
accuracy_fig = go.Figure()
accuracy_fig.add_trace(go.Scatter(
    x=accuracy_data['Model'],
    y=accuracy_data['Accuracy'],
    mode='lines+markers',
    name='Accuracy',
    line_shape='spline',
    line=dict(color='orange', width=3)
))
accuracy_fig.update_layout(
    title='Model Accuracy Comparison',
    xaxis_title='Model',
    yaxis_title='Accuracy',
    yaxis=dict(range=[0.84, 0.87], autorange=False),  # ✅ daha uygun aralık
    template='plotly_white'
)
tab_model.plotly_chart(accuracy_fig, use_container_width=True)

# 2. F1-Score Plot
f1_data = pd.DataFrame({
    'Model': results_df['Model'],
    'F1-Score': results_df['F1-Score']
})
f1_fig = go.Figure()
f1_fig.add_trace(go.Scatter(
    x=f1_data['Model'],
    y=f1_data['F1-Score'],
    mode='lines+markers',
    name='F1-Score',
    line_shape='spline',
    line=dict(color='green', width=3)
))
f1_fig.update_layout(
    title='Model F1-Score Comparison',
    xaxis_title='Model',
    yaxis_title='F1-Score',
    yaxis=dict(range=[0.53, 0.62], autorange=False),  # ✅ bu aralık net gösterir
    template='ggplot2'
)
tab_model.plotly_chart(f1_fig, use_container_width=True)

# 3. ROC AUC Plot
roc_auc_data = pd.DataFrame({
    'Model': results_df['Model'],
    'ROC AUC': results_df['ROC AUC']
})
roc_auc_fig = go.Figure()
roc_auc_fig.add_trace(go.Scatter(
    x=roc_auc_data['Model'],
    y=roc_auc_data['ROC AUC'],
    mode='lines+markers',
    name='ROC AUC',
    line_shape='spline',
    line=dict(color='red', width=3)
))
roc_auc_fig.update_layout(
    title='Model ROC AUC Comparison',
    xaxis_title='Model',
    yaxis_title='ROC AUC',
    yaxis=dict(range=[0.69, 0.73], autorange=False),  # ✅ en uygun aralık bu
    template='plotly_dark'
)
tab_model.plotly_chart(roc_auc_fig, use_container_width=True)

model = get_model()
#######################################################################################################################################################################
# tab_comment
#########################################################
text_1, text_2 = tab_comment.columns(2)

text_1.subheader(" - :red[What Metrics Mean:]",divider="gray")
text_1.markdown("""
* Accuracy: Shows how many of the model's predictions are correct.
* F1-Score: Shows the success of detecting churn. F1-Score is usually low.
* ROC AUC: Measures the model's ability to distinguish between churning and non-churning customers. The closer it is to 1, the more successful the model is.""")

text_1.subheader(" - :red[Interpretation of Results:]",divider="gray")
text_1.markdown("""
* GBM (Gradient Boosting Machine) model gives the best result in terms of both accuracy (86.45%) and F1-score.This model seems to be the most successful in detecting customer churn.
* Random Forest (RF), XGBoost and LightGBM models also give very close results and seem to be successful in churn prediction.
* CART (Decision Tree) model has the lowest success. It is weaker than other models in correctly predicting churning customers.
* ROC AUC values show that all models can make churn predictions at a certain level.""")

text_2.subheader(" - :red[Relationship with Churn:]",divider="gray")
text_2.markdown("""
* These models are used to predict when a customer will leave the company (churn). If the right model is selected, companies can prevent churn by offering special campaigns to these customers.
* Models with high F1-scores can better identify churning customers.
* Although accuracy is high, it is not enough on its own. Because the model may be able to predict customers who do not churn very well, but what is important is to be able to correctly predict those who do churn.
* Models with high ROC AUC values can better distinguish customer churn.""")

text_2.subheader(" - :red[Conclusion and Recommendations:]",divider="gray")
text_2.markdown("""
* One of the GBM, RF or LightGBM models should be preferred.
* Data processing techniques (for example, the SMOTE method to balance an unbalanced data set) can be used to better detect customer churn.
* The analysis can be expanded by adding different features (customer's last shopping date, complaint status, etc.) to further improve the model.""")

#######################################################################################################################################################################