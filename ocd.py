import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os


data_dir = r"D:\model\OCD_model"

#  Train/Test
train_file = f"{data_dir}/Y-BOCS_train.csv"
test_file  = f"{data_dir}/Y-BOCS_test.csv"

# قراءة البيانات
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

feature_cols = [col for col in train_df.columns if col.startswith("Q")]
X_train = train_df[feature_cols].astype(int)
y_train = train_df["Label"].astype(str)

X_test  = test_df[feature_cols].astype(int)
y_test  = test_df["Label"].astype(str)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

#  الموديل
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train_enc)

# التقييم
y_pred = clf.predict(X_test)
print(f"Y-BOCS Accuracy: {accuracy_score(y_test_enc, y_pred)*100:.2f}%")
print(classification_report(y_test_enc, y_pred, zero_division=0))

joblib.dump(clf, f"{data_dir}/Y-BOCS_model.pkl")
joblib.dump(le, f"{data_dir}/Y-BOCS_encoder.pkl")
json.dump({int(i): c for i, c in enumerate(le.classes_)},
          open(f"{data_dir}/Y-BOCS_map.json", "w"),
          indent=2)

print("Saved Y-BOCS model files in", data_dir)
