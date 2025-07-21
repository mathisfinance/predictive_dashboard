# prediction_co2.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- 1. Chargement des données ---
mois = [
    '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09',
    '2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09'
]

df = pd.DataFrame({
    'mois': pd.to_datetime(mois),
    'co2': [
        None, None, None, 298754.07, 255297.85, 333832.91, 234384.65, 325806.67, 323921.97, 185785.16, 158408.80, 374825.87,
        307984.09, 528830.13, 418137.15, 357420.76, 363507.50, 375950.13, 362419.48, 399446.80, 443985.07, 287034.27, None, None
    ],
    'intensite_km': [1, 3, 1, 2, 1, 3, 2, 3, 3, 1, 1, 2, 1, 3, 1, 2, 1, 3, 2, 3, 3, 1, 1, 2],
    'ca_mensuel': [
        456521.74, 1369565.22, 456521.74, 913043.48, 456521.74, 1369565.22, 913043.48, 1369565.22, 1369565.22, 456521.74, 456521.74, 913043.48,
        565217.39, 1695652.17, 565217.39, 1130434.78, 565217.39, 1695652.17, 1130434.78, 1695652.17, 1695652.17, 565217.39, 565217.39, 1130434.78
    ]
})

# --- 2. Visualisation ---
plt.figure(figsize=(10, 5))
plt.plot(df['mois'], df['co2'], marker='o', label='CO2 (kg)')
plt.title('Émissions de CO₂ vs CA mensuel')
plt.xlabel('Mois')
plt.ylabel('Valeurs')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3. Régression linéaire ---
train_df = df.dropna(subset=['co2'])
X_train = train_df[['ca_mensuel', 'intensite_km']]
y_train = train_df['co2']

model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction des mois manquants
mask_predict = df['mois'].dt.strftime('%Y-%m').isin(['2024-08', '2024-09'])
X_pred = df.loc[mask_predict, ['ca_mensuel', 'intensite_km']]
df.loc[mask_predict, 'co2_pred_lin'] = model.predict(X_pred)

# --- 4. Séries temporelles SARIMAX ---
ts = df.set_index('mois')['co2'].dropna()
model_ts = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_ts = model_ts.fit(disp=False)

forecast = results_ts.forecast(steps=2)
df.loc[mask_predict, 'co2_pred_sarimax'] = forecast.values

# --- 5. Affichage final des prédictions ---
print("\nPrédictions pour août et septembre 2024 :\n")
print(df[df['mois'].dt.strftime('%Y-%m').isin(['2024-08', '2024-09'])][['mois', 'co2_pred_lin', 'co2_pred_sarimax']])

# --- 6. Visualisation des prédictions pour août et septembre 2024 ---

# Courbe existante de CO₂
plt.figure(figsize=(10, 5))
plt.plot(df['mois'], df['co2'], label='CO₂ réel', marker='o')

# Ajout des prédictions linéaires
plt.plot(df['mois'], df['co2_pred_lin'], label='Prédiction linéaire', linestyle='--', marker='x')

# Ajout des prédictions SARIMAX
plt.plot(df['mois'], df['co2_pred_sarimax'], label='Prédiction SARIMAX', linestyle=':', marker='s')

plt.title("Prévisions CO₂ pour août et septembre 2024")
plt.xlabel("Mois")
plt.ylabel("Émissions de CO₂ (kg)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.to_excel("resultats_predictions_co2.xlsx", index=False)

plt.savefig("graphique_predictions_co2.png", dpi=300)
