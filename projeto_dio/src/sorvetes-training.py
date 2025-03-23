
# Importação das bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Carregar o dataset
print("Carregando os dados...")
sorvetes = pd.read_csv('venda_sorvete.csv')

# Separar features e labels
X = sorvetes[['Temperatura (°C)']].values
y = sorvetes['Quantidade Vendida'].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Definir hiperparâmetro de regularização
reg = 0.01

# Treinar um modelo de regressão logística
print("Treinando modelo com taxa de regularização de", reg)
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# Calcular a acurácia
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print("Acurácia:", acc)

# Calcular AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_scores[:, 1])
print("AUC:", str(auc))
