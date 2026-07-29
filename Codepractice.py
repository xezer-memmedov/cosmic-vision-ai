import os
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn .linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "exoplanets.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")


plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv(DATA_PATH)

print(df.head())
print("Dataset ölçüsü:", df.shape)
print("Sütunlar:", df.columns.tolist())
print("Boş dəyərlər:")
print(df.isnull().sum())

sns.countplot(data=df,x="habitable")

plt.title("Yaşanıla bilən planetlərin bölgüsü")
plt.xlabel("Habitable (0 = Xeyr, 1 = Bəli)")
plt.ylabel("Planet sayı")

plt.savefig(os.path.join(OUTPUT_DIR, "habitable_distribution.png"))
plt.show()

sns.scatterplot(
    data=df,
    x="distance_ly",
    y="equilibrium_temp_k",
    hue="habitable",
    s=100
)

plt.title("Məsafə və tarazlıq temperaturu")
plt.xlabel("Məsafə (işıq ili)")
plt.ylabel("Tarazlıq temperaturu (Kelvin)")

plt.savefig(os.path.join(OUTPUT_DIR, "temp_vs_distance.png"))
plt.show()

numeric_columns = [
    "distance_ly",
    "orbital_period_days",
    "planet_radius_earth",
    "planet_mass_earth",
    "star_temp_k",
    "equilibrium_temp_k",
    "habitable",
]

correlation = df[numeric_columns].corr()


sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Dəyişənlər arası korrelyasiya")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.show()


X = df[["distance_ly", "star_temp_k"]]
y = df["equilibrium_temp_k"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE
)

print("Təlim məlumatı:", X_train.shape)
print("Test məlumatı:", X_test.shape)

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

print("Əvvəlki X_train ölçüsü:", X_train.shape)
print("Yeni X_train_poly ölçüsü:", X_train_poly.shape)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

print("Faktiki dəyərlər:")
print(y_test.values)

print("Proqnozlaşdırılan dəyərlər:")
print(y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("RMSE:", round(rmse, 2), "K")
print("R²:", round(r2, 3))

plt.scatter(y_test, y_pred, color="blue")

# Mükəmməl proqnoz xətti
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--"
)

plt.title("Faktiki və proqnozlaşdırılan temperatur")
plt.xlabel("Faktiki temperatur (K)")
plt.ylabel("Proqnozlaşdırılan temperatur (K)")

plt.savefig(os.path.join(OUTPUT_DIR, "regression_result.png"))
plt.show()

feature_cols = [
    "distance_ly",
    "orbital_period_days",
    "planet_radius_earth",
    "planet_mass_earth",
    "star_temp_k",
    "equilibrium_temp_k"
]

X_class = df[feature_cols]
y_class = df["habitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X_class,
     y_class,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_class
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

logistic_model.fit(X_train_scaled, y_train)

y_pred_class = logistic_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_class)

print("Accuracy:", round(accuracy, 3))

print(classification_report(y_test, y_pred_class))

cm = confusion_matrix(y_test, y_pred_class)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Yaşanıla bilməyən", "Yaşanıla bilən"],
    yticklabels=["Yaşanıla bilməyən", "Yaşanıla bilən"]
)

plt.title("Confusion Matrix")
plt.xlabel("Modelin proqnozu")
plt.ylabel("Faktiki nəticə")

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

feature_importance = pd.Series(
    logistic_model.coef_[0],
    index=feature_cols
)

feature_importance = feature_importance.sort_values()

feature_importance.plot(
    kind="barh",
    color="green"

)

feature_importance.plot(
    kind="barh",
    color="green"
)

# 214-cü sətirdən buraya əlavə et
results_path = os.path.join(OUTPUT_DIR, "model_results.txt")

with open(results_path, "w", encoding="utf-8") as result_file:
    result_file.write("EXOPLANET ANALİZ NƏTİCƏLƏRİ\n")
    result_file.write("=" * 35 + "\n\n")
    result_file.write(f"Regression RMSE: {rmse:.2f} K\n")
    result_file.write(f"Regression R²: {r2:.3f}\n")
    result_file.write(f"Classification Accuracy: {accuracy:.3f}\n")

print("\n" + "=" * 50)
print("\n" + "=" * 50)
print("Exoplanet analizi tamamlandı!")
print("=" * 50)
print("Yaranan qrafiklər outputs qovluğunda saxlanıldı.")
