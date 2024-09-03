import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Veri setini yükleme
df = pd.read_csv("datasets/deprem-senaryosu-analiz-sonuclar.csv", delimiter=';', encoding='ISO-8859-1')

# Türkçe karakter düzeltme
replacements = [
    ('Ý', 'İ'),
    ('Þ', 'Ş'),
    ('Ð', 'Ğ')]

columns_to_replace = ['mahalle_adi', 'ilce_adi']
for column in columns_to_replace:
    if column in df.columns:
        for old_char, new_char in replacements:
            df[column] = df[column].replace(old_char, new_char, regex=True)

df.columns = [col.replace('_', ' ') for col in df.columns]
X = df[['cok agir hasarli bina sayisi']]
y = df['gecici barinma']


# 2. Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Tahminleri yap ve MSE değerini hesapla
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

model_filename = 'ibb_model.joblib'
joblib.dump(model,model_filename)
print(f'Model saved as {model_filename}')
