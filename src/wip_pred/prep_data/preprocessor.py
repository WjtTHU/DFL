from collections import Counter

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


class CalendarScaler:
    def __init__(self, cols: list[str]):
        self.cols = cols
        self.max_year_ = None

    def fit(self, X):
        if "year" in self.cols:
            self.max_year_ = X["year"].max()
        return self

    def transform(self, X):
        X = X.copy()
        if "year" in self.cols and self.max_year_ is not None:
            X["year"] = X["year"] / self.max_year_
        if "month" in self.cols:
            X["month"] = (X["month"] - 1) / 11
        if "day" in self.cols:
            X["day"] = (X["day"] - 1) / 30
        if "hour" in self.cols:
            X["hour"] = X["hour"] / 23
        if "minute" in self.cols:
            X["minute"] = X["minute"] / 59
        if "weekday" in self.cols:
            X["weekday"] = X["weekday"] / 6
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CategoryTokenizer:
    def __init__(self, min_freq=1, min_prop=0.0, unk_token="<UNK>", mask_token="<MASK>"):
        self.min_freq = min_freq
        self.min_prop = min_prop
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.freqs = None
        self.stoi = {}  # string-to-index
        self.itos = []  # index-to-string

    def fit(self, data):
        """
        data: list or pandas Series with categories
        """
        self.freqs = Counter(data)

        # initialize special tokens
        self.itos = [self.mask_token, self.unk_token]
        self.stoi = {self.mask_token: 0, self.unk_token: 1}

        # add categories with freq >= self.min_freq and prop >= self.min_prop
        for cat, freq in sorted(self.freqs.items()):
            if freq >= self.min_freq and freq / len(data) >= self.min_prop:
                if cat not in self.stoi:
                    idx = len(self.itos)
                    self.stoi[cat] = idx
                    self.itos.append(cat)

    def transform(self, data):
        """from category to idx; returns pd.Series"""
        if not isinstance(data, (list, pd.Series)):
            raise TypeError("data must be a list or pandas Series")
        encoded = [self.stoi.get(cat, self.stoi[self.unk_token]) for cat in data]
        return pd.Series(encoded, index=data.index if isinstance(data, pd.Series) else None)

    def inverse_transform(self, ids):
        """from index to category; returns pd.Series"""
        if not isinstance(ids, (list, pd.Series)):
            raise TypeError("ids must be a list or pandas Series")
        decoded = [self.itos[i] if i < len(self.itos) else self.unk_token for i in ids]
        return pd.Series(decoded, index=ids.index if isinstance(ids, pd.Series) else None)

    def vocab_size(self):
        return len(self.itos)

    def vocab(self):
        return self.stoi.copy()


class Preprocessor:
    def __init__(
        self,
        datetime_col: str,
        temporal_features: list[str] | None = None,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        min_cat_freq: int = 3,
        min_cat_prop: float = 0.0,
    ) -> None:
        temporal_features = temporal_features or [
            "year",
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
        ]
        numeric_cols = numeric_cols or []
        categorical_cols = categorical_cols or []

        assert len(temporal_features) == len(set(temporal_features))
        assert set(temporal_features).issubset(
            set(["year", "month", "day", "weekday", "hour", "minute"])
        )

        self.datetime_col = datetime_col
        self.temporal_features = temporal_features
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        self.calender_scaler = CalendarScaler(temporal_features)
        self.scaler = StandardScaler() if numeric_cols else None
        self.label_encoders = dict()

        for col in categorical_cols:
            self.label_encoders[col] = CategoryTokenizer(
                min_freq=min_cat_freq, min_prop=min_cat_prop
            )

    def create_temporal_features(self, df: DataFrame):
        for temporal_feat in self.temporal_features:  # create columns of temporal features
            if temporal_feat == "year":
                df["year"] = df[self.datetime_col].dt.year
            elif temporal_feat == "month":
                df["month"] = df[self.datetime_col].dt.month
            elif temporal_feat == "day":
                df["day"] = df[self.datetime_col].dt.day
            elif temporal_feat == "weekday":
                df["weekday"] = df[self.datetime_col].dt.weekday
            elif temporal_feat == "hour":
                df["hour"] = df[self.datetime_col].dt.hour
            elif temporal_feat == "minute":
                df["minute"] = df[self.datetime_col].dt.minute

        return df

    def fit(self, df: DataFrame):
        if self.scaler is not None:
            self.scaler.fit(df[self.numeric_cols])

        for col, label_encoder in self.label_encoders.items():
            label_encoder.fit(df[col])

        self.calender_scaler.fit(df[self.temporal_features])

    def transform(self, df: DataFrame):
        if self.scaler is not None:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        for col, label_encoder in self.label_encoders.items():
            df[col] = label_encoder.transform(df[col])

        df[self.temporal_features] = self.calender_scaler.transform(df[self.temporal_features])

        return df
