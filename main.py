import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset

# 1. Load AG News Dataset
print("Loading dataset...")
dataset = load_dataset("wangrongsheng/ag_news")
df = pd.DataFrame(dataset["train"])

# 2. Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing text...")
df["text"] = df["text"].apply(clean_text)

# 3. TF-IDF Vectorization
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Logistic Regression
print("Training Logistic Regression model...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

# 6. Decision Tree
print("Training Decision Tree model...")
dt = DecisionTreeClassifier(max_depth=20, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Results")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

# 7. Plot Decision Tree (top 3 levels only for readability)
print("Rendering Decision Tree (top 3 levels)...")
plt.figure(figsize=(25, 12))
plot_tree(
    dt,
    max_depth=3,
    feature_names=vectorizer.get_feature_names_out(),
    class_names=["World", "Sports", "Business", "Sci/Tech"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (Depth 3)")
plt.tight_layout()
plt.savefig("decision_tree_plot.png")  # Saves to file
plt.show()
