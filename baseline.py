import matplotlib.pyplot as plt
import numpy as np
from sarcasm import Sarcasm
from preprocess import SarcasmDataManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)



class SarcasmBaseline:
    def __init__(self):
        self.data_manager = SarcasmDataManager(Sarcasm())

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="liblinear"
        )

    def load_data(self):
        train_df, validation_df, test_df = self.data_manager.prepare_dataframes()

        X_train, y_train = self.data_manager.get_features_and_labels(train_df)
        X_val, y_val = self.data_manager.get_features_and_labels(validation_df)
        X_test, y_test = self.data_manager.get_features_and_labels(test_df)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def vectorize_data(self, X_train, X_val, X_test):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        X_test_vec = self.vectorizer.transform(X_test)

        return X_train_vec, X_val_vec, X_test_vec

    def train(self, X_train_vec, y_train):
        self.model.fit(X_train_vec, y_train)

    def evaluate_split(self, X_vec, y_true, split_name="Validation"):
        y_pred = self.model.predict(X_vec)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        print(f"\n{split_name} Results")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print(f"\n{split_name} Classification Report")
        print("-" * 40)
        print(classification_report(y_true, y_pred, zero_division=0))

        return y_pred, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def plot_confusion_matrix(self, y_true, y_pred, split_name="Validation"):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

        disp.plot(cmap="Blues")
        plt.title(f"{split_name} Confusion Matrix")
        plt.show()

    def show_top_features(self, top_n=20):
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coefficients = self.model.coef_[0]

        top_positive_idx = np.argsort(coefficients)[-top_n:]
        top_negative_idx = np.argsort(coefficients)[:top_n]

        print("\nTop features predicting sarcasm (label=1)")
        print("-" * 40)
        for idx in reversed(top_positive_idx):
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

        print("\nTop features predicting non-sarcasm (label=0)")
        print("-" * 40)
        for idx in top_negative_idx:
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

    def run(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        X_train_vec, X_val_vec, X_test_vec = self.vectorize_data(X_train, X_val, X_test)
        self.train(X_train_vec, y_train)

        val_preds, val_metrics = self.evaluate_split(X_val_vec, y_val, split_name="Validation")
        self.plot_confusion_matrix(y_val, val_preds, split_name="Validation")
        self.print_evaluation_summary(val_metrics, split_name="Validation")

        test_preds, test_metrics = self.evaluate_split(X_test_vec, y_test, split_name="Test")
        self.plot_confusion_matrix(y_test, test_preds, split_name="Test")
        self.print_evaluation_summary(test_metrics, split_name="Test")

        self.show_top_features(top_n=20)

        return {
            "validation": val_metrics,
            "test": test_metrics
        }
    
    def print_evaluation_summary(self, metrics, split_name="Test"):
        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        print(f"\n{split_name} Evaluation Summary")
        print("-" * 40)
        print(f"The baseline TF-IDF + Logistic Regression model has been evaluated on the {split_name.lower()} set.")
        print(f"The {split_name.lower()} accuracy is {accuracy:.4f}.")
        print(f"The {split_name.lower()} precision is {precision:.4f}.")
        print(f"The {split_name.lower()} recall is {recall:.4f}.")
        print(f"The {split_name.lower()} F1-score is {f1:.4f}.")


if __name__ == "__main__":
    baseline = SarcasmBaseline()
    results = baseline.run()

    print("\nFinal Metrics Summary")
    print("-" * 40)
    print(results)