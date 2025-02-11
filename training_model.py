import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils import preprocess_text


df = pd.read_csv("flu_social_media_dataset.csv")

df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
label_mapping = {'non-flu': 0, 'flu': 1}
df['Label_Num'] = df['Label'].map(label_mapping)


vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english')
X = vectorizer.fit_transform(df['Cleaned_Text']).toarray()
y = df['Label_Num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


log_reg = LogisticRegression(C=0.01, penalty='l2', solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)


rf_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation
def evaluate_model(name, model, y_true, y_pred, X_test):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_true, model.predict_proba(X_test)[:, 1]))


cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy (Logistic Regression): {cv_scores.mean():.4f}")


evaluate_model("Logistic Regression", log_reg, y_test, y_pred_lr, X_test)
evaluate_model("Random Forest", rf_model, y_test, y_pred_rf, X_test)


joblib.dump(log_reg, "model_files/flu_detection_log_reg_model.pkl")
joblib.dump(vectorizer, "model_files/flu_vectorizer.pkl")
