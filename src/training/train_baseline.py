import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc

# Load data (you will download creditcard.csv into data/raw/)
df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print("PR-AUC:", round(pr_auc, 4))

y_pred = (y_probs >= 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
