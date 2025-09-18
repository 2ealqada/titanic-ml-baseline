import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # geht ein Ordner hoch (src -> titanic-ml-baseline)
DATA_DIR = os.path.join(BASE_DIR, "data")

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))


print(train.head())
print("......")
print(test.head())



# 1. Überblick über fehlende Werte
print(train.isna().sum())

# 2. Verteilung Zielvariable
sns.countplot(data=train, x="Survived")
plt.show()

# 3. Überleben nach Geschlecht
sns.countplot(data=train, x="Sex", hue="Survived")
plt.show()

# 4. Überleben nach Klasse (Pclass)
sns.countplot(data=train, x="Pclass", hue="Survived")
plt.show()

# 5. Altersverteilung
sns.histplot(data=train, x="Age", bins=20, kde=True, hue="Survived")
plt.show()
sns.boxplot(data=train, x="Survived", y="Fare")
plt.show()

