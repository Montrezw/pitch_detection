import re
import string
import pandas as pd


class SarcasmDataManager:
    def __init__(self, dataset_builder):
        self.dataset_builder = dataset_builder

        self.text_column = "comments"
        self.label_column = "contains_slash_s"
        self.lowercase = True
        self.remove_urls = True
        self.remove_mentions = False
        self.remove_punctuation = False
        self.strip_whitespace = True
        self.dataset_train = None
        self.dataset_validation = None
        self.dataset_test = None
        self.train_df = None
        self.validation_df = None
        self.test_df = None

    def load(self):
        self.dataset_builder.download_and_prepare()
        self.dataset_train = self.dataset_builder.as_dataset(split="train")
        self.dataset_validation = self.dataset_builder.as_dataset(split="validation")
        self.dataset_test = self.dataset_builder.as_dataset(split="test")

        return self.dataset_train, self.dataset_validation, self.dataset_test

    def dataset_to_dataframe(self, dataset_split):
        return pd.DataFrame(dataset_split)

    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.strip_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess_dataframe(self, df):
        df = df.copy()

        df = df.dropna(subset=[self.text_column, self.label_column])

        df[self.text_column] = df[self.text_column].astype(str).apply(self.clean_text)
        df[self.label_column] = df[self.label_column].astype(int)

        df = df[df[self.text_column].str.len() > 0].reset_index(drop=True)

        return df

    def prepare_dataframes(self):
        if self.dataset_train is None:
            self.load()

        self.train_df = self.preprocess_dataframe(
            self.dataset_to_dataframe(self.dataset_train)
        )
        self.validation_df = self.preprocess_dataframe(
            self.dataset_to_dataframe(self.dataset_validation)
        )
        self.test_df = self.preprocess_dataframe(
            self.dataset_to_dataframe(self.dataset_test)
        )

        return self.train_df, self.validation_df, self.test_df

    def get_features_and_labels(self, df):
        X = df[self.text_column]
        y = df[self.label_column]
        return X, y

    def print_summary(self):
        if self.train_df is None:
            raise ValueError("Call prepare_dataframes() first.")

        print("Dataset Summary")
        print(f"Train: {len(self.train_df)}")
        print(f"Validation: {len(self.validation_df)}")
        print(f"Test: {len(self.test_df)}")

        print("\nTrain label distribution:")
        print(self.train_df[self.label_column].value_counts())

        print("\nValidation label distribution:")
        print(self.validation_df[self.label_column].value_counts())

        print("\nTest label distribution:")
        print(self.test_df[self.label_column].value_counts())