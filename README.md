import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier

# Veriyi yükleimport
df = pd.read_csv('/content/healthcare-dataset-stroke-data.csv')

# Veriye ilk bakış
display(df.head())
print(df.info())
# Eksik değerleri temizle (bmi sütununda eksikler olabilir)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Gereksiz sütunları kaldır (id gibi)
df = df.drop(columns=['id'])

# Kategorik verileri sayısal verilere dönüştür
le = LabelEncoder()
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Eğitim ve test setlerine ayır
X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)
# Model Eğitimi (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred = model.predict(X_test)

print("Model Doğruluğu:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


# Lojistik Regresyon Modelini Eğitme
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred_log = log_model.predict(X_test)

print("Logistic Regression Model Doğruluğu:", accuracy_score(y_test, y_pred_log))
print("\nLogistic Regression Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_log))

# Karar Ağacı Modelini Eğitme
dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred_dt = dtree_model.predict(X_test)

print("Decision Tree Model Doğruluğu:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_dt))

# KNN Modelini Eğitme (varsayılan k=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred_knn = knn_model.predict(X_test)

print("KNN Model Doğruluğu:", accuracy_score(y_test, y_pred_knn))
print("\nKNN Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_knn))

# Gradient Boosting Modelini Eğitme
gbc_model = GradientBoostingClassifier(random_state=42)
gbc_model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred_gbc = gbc_model.predict(X_test)

print("Gradient Boosting Model Doğruluğu:", accuracy_score(y_test, y_pred_gbc))
print("\nGradient Boosting Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_gbc))

# XGBoost Modelini Eğitme
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Model Doğruluğu:", accuracy_score(y_test, y_pred_xgb))
print("\nXGBoost Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_xgb))
