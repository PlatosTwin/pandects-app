"""
M&A Agreement Classification for SEC Filings.

This module provides a classifier to determine whether an SEC filing text
represents an M&A agreement.

Notes:
- Supervised training requires both positive and negative examples.
- Use at least 10 examples; 100+ yields more stable behavior for similarity features.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false

import re
import warnings
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from scipy.sparse import spmatrix


class ExhibitClassifier:
    """
    Supervised classifier for identifying M&A agreements from SEC filing text.

    Features extracted include:
    - Document structure indicators (sections, articles, exhibits)
    - Legal language patterns specific to M&A agreements
    - Term frequencies of M&A-specific vocabulary
    - Document length and complexity metrics
    - Semantic similarity to known M&A agreements

    Example:
        ```python
        # Train the classifier with labeled agreement texts
        classifier = ExhibitClassifier()
        classifier.fit(texts, labels=labels)

        # Classify a new SEC filing
        probability = classifier.predict_proba("SEC filing text...")
        is_ma_agreement = classifier.predict("SEC filing text...")
        ```
    """

    def __init__(
        self,
        max_features: int = 5000,
        char_max_features: int = 3000,
        word_ngram_range: tuple[int, int] = (1, 3),
        char_ngram_range: tuple[int, int] = (3, 5),
        logreg_c: float = 1.0,
        logreg_max_iter: int = 1000,
        class_weight: str | dict[int, float] | None = "balanced",
        start_scan_chars: int = 2000,
        random_state: int = 42,
    ):
        """
        Initialize the M&A agreement classifier.

        Args:
            max_features: Maximum number of TF-IDF features to use
            random_state: Random seed for reproducibility
        """
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.logreg_c = logreg_c
        self.logreg_max_iter = logreg_max_iter
        self.class_weight = class_weight
        self.start_scan_chars = start_scan_chars
        self.random_state = random_state

        # Initialize hashing-based text feature pipeline.
        self.tfidf_vectorizer = HashingVectorizer(
            n_features=max_features,
            stop_words="english",
            ngram_range=word_ngram_range,
            alternate_sign=False,
            norm="l2",
        )
        self.char_tfidf_vectorizer = HashingVectorizer(
            n_features=char_max_features,
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            alternate_sign=False,
            norm="l2",
        )
        self.word_tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        self.char_tfidf_transformer = TfidfTransformer(sublinear_tf=True)

        self.feature_scaler = StandardScaler(with_mean=False)
        self.is_fitted = False
        self.training_texts: list[str] = []  # Store for similarity computation
        self.training_tfidf_matrix: spmatrix | None = None
        self.binary_classifier: LogisticRegression | None = None
        # Decision threshold for predict(); set by train/tune and persisted in joblib.
        self.decision_threshold: float = 0.5

        # M&A-specific vocabulary patterns
        self.ma_keywords = [
            "merger",
            "acquisition",
            "merger agreement",
            "purchase agreement",
            "stock purchase",
            "asset purchase",
            "merger consideration",
            "surviving corporation",
            "constituent corporations",
            "merger sub",
            "tender offer",
            "exchange ratio",
            "merger closing",
            "effective time",
            "dissenting shares",
            "appraisal rights",
            "cash merger",
            "stock merger",
            "reverse merger",
            "triangular merger",
            "short form merger",
            "representations and warranties",
            "covenants",
            "closing conditions",
            "material adverse change",
            "mac",
            "material adverse effect",
            "mae",
            "termination fee",
            "break up fee",
            "collar",
            "walk away rights",
            "fairness opinion",
            "solvency opinion",
            "proxy statement",
            "definitive agreement",
            "letter of intent",
            "due diligence",
            "antitrust clearance",
            "hsr act",
            "regulatory approval",
            "stockholder approval",
            "board recommendation",
            "go shop",
            "no shop",
            "matching rights",
            "superior proposal",
        ]
        self.start_agreement_title_phrases = [
            "agreement and plan of amalgamation",
            "agreement and plan of merger",
            "agreement and plan of reorganization",
            "agreement of merger",
            "asset purchase agreement",
            "business combination agreement",
            "equity purchase agreement",
            "interest purchase agreement",
            "membership interest purchase agreement",
            "merger agreement",
            "plan of merger",
            "purchase and sale agreement",
            "securities purchase agreement",
            "share purchase agreement",
            "stock purchase agreement",
            "stock purchase and sale agreement",
        ]

        self.ma_hard_negative_phrases = [
            "agreement of limited partnership",
            "amended and restated",
            "amendment",
            "co-operation agreement",
            "commercial lease",
            "compensation",
            "confidentiality agreement",
            "consulting agreement",
            "contractor agreement",
            "convertible note",
            "cooperation agreement",
            "covenant agreement",
            "credit agreement",
            "deferred compensation",
            "distribution agreement",
            "employee stock",
            "employment agreement",
            "equity plan",
            "exchange agreement",
            "exhibit",
            "general release of claims",
            "guaranty",
            "incentive",
            "indemnification agreement",
            "indemnity agreement",
            "investment management trust agreement",
            "joint venture agreement",
            "lease agreement",
            "letter agreement",
            "letter of intent",
            "license agreement",
            "limited partnership agreement",
            "loan agreement",
            "loan and security agreement",
            "management agreement",
            "master lease",
            "memorandum agreement",
            "memorandum of understanding",
            "non-competition",
            "non-solicitation agreement",
            "note purchase agreement",
            "option",
            "performance",
            "please see pdf version",
            "private placement",
            "promissory note",
            "real estate",
            "registration rights",
            "restricted shares",
            "restricted stock",
            "retirement plan",
            "royalty",
            "schedule",
            "securities subscription agreement",
            "secured note",
            "senior secured",
            "separation",
            "separation and general release agreement",
            "service agreement",
            "services agreement",
            "shareholder rights agreements",
            "side letter agreement",
            "sponsor agreement",
            "sponsorship agreement",
            "subscription agreement",
            "supply agreement",
            "support agreement",
            "support letter",
            "term loan",
            "termination agreement",
            "transition services",
            "underwriting agreement",
            "voting agreement",
            "warrant",
        ]

    @staticmethod
    def _normalize_for_feature_scan(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _extract_document_features(self, text: str) -> np.ndarray:
        """
        Extract document-level features specific to M&A agreements.

        Args:
            text: Raw text content of the SEC filing

        Returns:
            Feature vector as numpy array
        """
        # Normalize text
        text = str(text).lower()

        # Use first N chars as a proxy for the first page/start of document
        first_chunk = text[: self.start_scan_chars]
        first_chunk_norm = self._normalize_for_feature_scan(first_chunk)

        # Basic document statistics
        words = text.split()
        num_words = len(words)
        first_chunk_len = len(first_chunk_norm)

        title_positions = [
            first_chunk_norm.find(phrase)
            for phrase in self.start_agreement_title_phrases
            if first_chunk_norm.find(phrase) >= 0
        ]
        has_start_title_hit = len(title_positions) > 0
        title_pos = min(title_positions) if has_start_title_hit else -1
        start_has_agreement_title = 1.0 if has_start_title_hit else 0.0
        start_has_agreement_title_pos = (
            float(title_pos) / max(first_chunk_len, 1) if has_start_title_hit else 0.0
        )

        hard_negative_positions = [
            first_chunk_norm.find(phrase)
            for phrase in self.ma_hard_negative_phrases
            if first_chunk_norm.find(phrase) >= 0
        ]
        has_start_hard_negative_hit = len(hard_negative_positions) > 0
        hard_negative_pos = (
            min(hard_negative_positions) if has_start_hard_negative_hit else -1
        )
        start_hard_negative = 1.0 if has_start_hard_negative_hit else 0.0
        start_hard_negative_pos = (
            float(hard_negative_pos) / max(first_chunk_len, 1)
            if has_start_hard_negative_hit
            else 0.0
        )

        # M&A-specific language patterns
        ma_keyword_count = sum(1 for keyword in self.ma_keywords if keyword in text)
        ma_keyword_density = ma_keyword_count / num_words if num_words > 0 else 0.0

        # Compile feature vector
        features = [
            num_words,
            ma_keyword_count,
            ma_keyword_density,
            start_has_agreement_title,
            start_has_agreement_title_pos,
            start_hard_negative,
            start_hard_negative_pos,
        ]

        return np.array(features, dtype=float)

    def extract_document_features(self, text: str) -> np.ndarray:
        return self._extract_document_features(text)

    def extract_document_features_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self._extract_document_features(text) for text in texts])

    def _compute_similarity_features(self, text: str) -> np.ndarray:
        """
        Compute semantic similarity to training M&A agreements.

        Args:
            text: Text to compute similarity for

        Returns:
            Similarity features as numpy array
        """
        if not self.training_texts:
            return np.array([0.0, 0.0, 0.0])  # No training data yet

        if self.training_tfidf_matrix is None:
            self.training_tfidf_matrix = self._build_text_matrix(self.training_texts)

        query_vector = self._build_text_matrix([text])
        training_vectors = self.training_tfidf_matrix

        similarities = cosine_similarity(query_vector, training_vectors).flatten()

        # Summary statistics
        max_similarity = np.max(similarities)
        mean_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)

        return np.array([max_similarity, mean_similarity, median_similarity])

    def _compute_similarity_features_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3))
        if not self.training_texts:
            return np.zeros((len(texts), 3))

        if self.training_tfidf_matrix is None:
            self.training_tfidf_matrix = self._build_text_matrix(self.training_texts)

        query_matrix = self._build_text_matrix(texts)
        training_vectors = self.training_tfidf_matrix

        similarities = cosine_similarity(query_matrix, training_vectors)
        max_similarity = np.max(similarities, axis=1)
        mean_similarity = np.mean(similarities, axis=1)
        median_similarity = np.median(similarities, axis=1)
        return np.vstack([max_similarity, mean_similarity, median_similarity]).T

    def _build_text_matrix(self, texts: list[str]) -> spmatrix:
        word_matrix = cast(spmatrix, self.tfidf_vectorizer.transform(texts))
        char_matrix = cast(spmatrix, self.char_tfidf_vectorizer.transform(texts))
        word_matrix = cast(spmatrix, self.word_tfidf_transformer.transform(word_matrix))
        char_matrix = cast(spmatrix, self.char_tfidf_transformer.transform(char_matrix))
        return cast(spmatrix, sparse_hstack([word_matrix, char_matrix]))

    @staticmethod
    def compute_training_similarity_features(tfidf_matrix: spmatrix) -> np.ndarray:
        """Compute similarity stats for each training example (exclude self)."""
        if tfidf_matrix.shape[0] <= 1:
            return np.zeros((tfidf_matrix.shape[0], 3))

        sims = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sims, np.nan)
        max_sim = np.nanmax(sims, axis=1)
        mean_sim = np.nanmean(sims, axis=1)
        median_sim = np.nanmedian(sims, axis=1)
        return np.vstack([max_sim, mean_sim, median_sim]).T

    def fit(
        self, texts: list[str], labels: list[int] | None = None
    ) -> "ExhibitClassifier":
        """
        Train the classifier on M&A agreement texts.

        Args:
            texts: List of M&A agreement text content
            labels: List of 0/1 labels for supervised training

        Returns:
            Self for method chaining
        """
        if len(texts) < 10:
            warnings.warn(
                f"Training with only {len(texts)} examples. Consider using more training data for better performance."
            )

        print(f"Training M&A classifier with {len(texts)} agreements...")

        if labels is None or not any(label == 0 for label in labels):
            raise RuntimeError(
                "Supervised training requires both positive and negative labels."
            )

        # Store training texts for similarity computation
        self.training_texts = texts.copy()

        # Fit hashing TF-IDF transformers.
        print("- Fitting hash TF-IDF transformers...")
        word_counts = cast(spmatrix, self.tfidf_vectorizer.transform(texts))
        char_counts = cast(spmatrix, self.char_tfidf_vectorizer.transform(texts))
        _ = self.word_tfidf_transformer.fit(word_counts)
        _ = self.char_tfidf_transformer.fit(char_counts)

        # Extract document-level features
        print("- Extracting document-level features...")
        doc_features = np.array(
            [self._extract_document_features(text) for text in texts]
        )

        # Extract TF-IDF features (use smaller subset for efficiency)
        print("- Extract TF-IDF features...")
        tfidf_matrix = self._build_text_matrix(texts)
        self.training_tfidf_matrix = tfidf_matrix

        similarity_features = self.compute_training_similarity_features(tfidf_matrix)

        # Combine all features
        doc_sparse = csr_matrix(doc_features)
        similarity_sparse = csr_matrix(similarity_features)
        all_features = cast(
            spmatrix, sparse_hstack([doc_sparse, tfidf_matrix, similarity_sparse])
        )

        # Scale features
        print("- Scale all features...")
        _ = self.feature_scaler.fit(all_features)
        scaled_features = self.feature_scaler.transform(all_features)

        print("Fit model...")
        y = np.array(labels, dtype=int)
        self.binary_classifier = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=self.logreg_max_iter,
            C=self.logreg_c,
            random_state=self.random_state,
        )
        _ = self.binary_classifier.fit(scaled_features, y)

        self.is_fitted = True
        print("Classifier trained successfully using logistic regression")

        return self

    def predict_proba(self, text: str) -> float:
        """
        Predict the probability that the text is an M&A agreement.

        Args:
            text: SEC filing text content

        Returns:
            Probability score between 0 and 1 (higher = more likely M&A agreement)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")

        # Extract features
        doc_features = self._extract_document_features(text)
        tfidf_matrix = self._build_text_matrix([text])
        similarity_features = self._compute_similarity_features(text)

        # Combine features
        doc_sparse = csr_matrix([doc_features])
        similarity_sparse = csr_matrix([similarity_features])
        all_features = cast(
            spmatrix, sparse_hstack([doc_sparse, tfidf_matrix, similarity_sparse])
        )
        scaled_features = self.feature_scaler.transform(all_features)

        if self.binary_classifier is None:
            raise RuntimeError("Classifier must be fitted before making predictions")

        probability = self.binary_classifier.predict_proba(scaled_features)[0, 1]
        return float(np.clip(probability, 0.0, 1.0))

    def predict_proba_batch(self, texts: list[str]) -> list[float]:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")
        if not texts:
            return []

        doc_features = np.array(
            [self._extract_document_features(text) for text in texts]
        )
        tfidf_matrix = self._build_text_matrix(texts)
        similarity_features = self._compute_similarity_features_batch(texts)

        doc_sparse = csr_matrix(doc_features)
        similarity_sparse = csr_matrix(similarity_features)
        all_features = cast(
            spmatrix, sparse_hstack([doc_sparse, tfidf_matrix, similarity_sparse])
        )
        scaled_features = self.feature_scaler.transform(all_features)

        if self.binary_classifier is None:
            raise RuntimeError("Classifier must be fitted before making predictions")

        probabilities = self.binary_classifier.predict_proba(scaled_features)[:, 1]
        return [float(p) for p in np.clip(probabilities, 0.0, 1.0)]

    def predict_batch(
        self, texts: list[str], threshold: float | None = None
    ) -> list[bool]:
        threshold_value = self.decision_threshold if threshold is None else threshold
        probabilities = self.predict_proba_batch(texts)
        return [prob >= threshold_value for prob in probabilities]

    def predict(self, text: str, threshold: float | None = None) -> bool:
        """
        Predict whether the text is an M&A agreement.

        Args:
            text: SEC filing text content
            threshold: Probability threshold for classification. If None, use decision_threshold.

        Returns:
            True if predicted to be M&A agreement, False otherwise
        """
        threshold_value = self.decision_threshold if threshold is None else threshold
        probability = self.predict_proba(text)
        return probability >= threshold_value

    def save(self, filepath: str | Path) -> None:
        """
        Save the trained classifier to disk.

        Args:
            filepath: Path to save the classifier
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")

        path = Path(filepath)

        # Save using joblib
        model_data = {
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "char_tfidf_vectorizer": self.char_tfidf_vectorizer,
            "word_tfidf_transformer": self.word_tfidf_transformer,
            "char_tfidf_transformer": self.char_tfidf_transformer,
            "feature_scaler": self.feature_scaler,
            "training_texts": self.training_texts,
            "binary_classifier": self.binary_classifier,
            "max_features": self.max_features,
            "char_max_features": self.char_max_features,
            "word_ngram_range": self.word_ngram_range,
            "char_ngram_range": self.char_ngram_range,
            "logreg_c": self.logreg_c,
            "logreg_max_iter": self.logreg_max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "ma_keywords": self.ma_keywords,
            "start_agreement_title_phrases": self.start_agreement_title_phrases,
            "ma_hard_negative_phrases": self.ma_hard_negative_phrases,
            "start_scan_chars": self.start_scan_chars,
            "decision_threshold": getattr(self, "decision_threshold", 0.5),
        }

        _ = joblib.dump(model_data, path)
        print(f"Classifier saved to {path}")

    @classmethod
    def load(cls, filepath: str | Path) -> "ExhibitClassifier":
        """
        Load a trained classifier from disk.

        Args:
            filepath: Path to the saved classifier

        Returns:
            Loaded classifier instance
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Classifier file not found: {path}")

        # Load model data
        model_data = joblib.load(path)

        # Create new instance
        classifier = cls(
            max_features=model_data["max_features"],
            char_max_features=model_data["char_max_features"],
            word_ngram_range=model_data["word_ngram_range"],
            char_ngram_range=model_data["char_ngram_range"],
            logreg_c=model_data["logreg_c"],
            logreg_max_iter=model_data["logreg_max_iter"],
            class_weight=model_data["class_weight"],
            start_scan_chars=model_data["start_scan_chars"],
            random_state=model_data["random_state"],
        )

        # Restore trained components (must use saved vectorizers so feature dims match the saved scaler/LR)
        classifier.tfidf_vectorizer = model_data["tfidf_vectorizer"]
        classifier.char_tfidf_vectorizer = model_data["char_tfidf_vectorizer"]
        classifier.word_tfidf_transformer = model_data["word_tfidf_transformer"]
        classifier.char_tfidf_transformer = model_data["char_tfidf_transformer"]
        classifier.feature_scaler = model_data["feature_scaler"]
        classifier.training_texts = model_data["training_texts"]
        # Recompute from training_texts so dimensions match _build_text_matrix.
        classifier.training_tfidf_matrix = None
        classifier.binary_classifier = model_data["binary_classifier"]
        classifier.ma_keywords = model_data["ma_keywords"]
        classifier.start_agreement_title_phrases = model_data[
            "start_agreement_title_phrases"
        ]
        classifier.ma_hard_negative_phrases = model_data["ma_hard_negative_phrases"]
        classifier.start_scan_chars = int(model_data["start_scan_chars"])
        classifier.decision_threshold = model_data["decision_threshold"]
        classifier.is_fitted = True

        # Sanity check: pipeline output dim must match what the scaler was fitted on
        sample = (
            classifier.training_texts[0]
            if classifier.training_texts
            else "agreement merger acquisition consideration closing"
        )
        doc_dim = len(classifier._extract_document_features(sample))
        tfidf_cols = classifier._build_text_matrix([sample]).shape[1]
        pipeline_n_features = doc_dim + tfidf_cols + 3
        scaler_n_features = getattr(classifier.feature_scaler, "n_features_in_", None)
        if scaler_n_features is not None and pipeline_n_features != scaler_n_features:
            raise ValueError(
                (
                    f"Exhibit classifier model file is inconsistent: feature_scaler was fitted on "
                    f"{scaler_n_features} features but the loaded vectorizers and pipeline produce "
                    f"{pipeline_n_features}. This usually means the artifact was saved with an older or "
                    f"different feature pipeline. Retrain and save a new model, e.g. run "
                    f"exhibit_classifier_train.py (train or tune mode) and point the staging asset/CLI "
                    f"at the new .joblib file."
                )
            )

        print(f"Classifier loaded from {path}")
        return classifier

    def get_feature_names(self) -> list[str]:
        """Get the list of feature names used by the classifier."""
        doc_feature_names = [
            "num_words",
            "ma_keyword_count",
            "ma_keyword_density",
            "start_has_agreement_title",
            "start_has_agreement_title_pos",
            "start_hard_negative",
            "start_hard_negative_pos",
        ]

        word_vocab = [f"hash_word:{i}" for i in range(self.max_features)]
        char_vocab = [f"hash_char:{i}" for i in range(self.char_max_features)]

        # Similarity features
        similarity_feature_names = [
            "max_similarity",
            "mean_similarity",
            "median_similarity",
        ]

        return doc_feature_names + word_vocab + char_vocab + similarity_feature_names

    def get_model_coefficients(self) -> dict[str, float] | None:
        """
        Get the coefficients of the linear model (if supervised).

        Returns:
            Dictionary of feature name -> coefficient, or None if not applicable.
        """
        if not self.is_fitted or self.binary_classifier is None:
            return None

        if hasattr(self.binary_classifier, "coef_"):
            coefs = self.binary_classifier.coef_[0]
            names = self.get_feature_names()

            if len(coefs) != len(names):
                # Fallback if dimensions don't match (e.g. SVD components changed)
                return {f"feature_{i}": float(c) for i, c in enumerate(coefs)}

            return dict(zip(names, [float(c) for c in coefs]))

        return None

    def get_feature_importance(self, text: str) -> dict[str, float]:
        """
        Get feature values for a given text (for interpretation).

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping feature names to their values
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before feature analysis")

        # Extract features
        doc_features = self._extract_document_features(text)
        similarity_features = self._compute_similarity_features(text)
        tfidf_matrix = self._build_text_matrix([text])
        tfidf_values = cast(csr_matrix, tfidf_matrix).toarray()[0]

        all_feature_values = np.hstack(
            [doc_features, tfidf_values, similarity_features]
        )
        all_feature_names = self.get_feature_names()

        return dict(zip(all_feature_names, all_feature_values))


def load_training_data(
    data_path: str,
) -> tuple[list[str], list[int] | None, list[str] | None]:
    """
    Load M&A agreement texts from a data file.

    This is a helper function to load training data. The format depends on
    how your M&A agreement data is stored.

    Args:
        data_path: Path to the training data file

    Returns:
        Tuple of text list, optional label list, and optional URL list
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        # Assume there's a 'text' column with the agreement content
        if "text" in df.columns:
            texts = df["text"].fillna("").astype(str).tolist()
            labels = None
            if "label" in df.columns:
                if bool(df["label"].isna().any()):
                    raise ValueError(
                        "CSV file contains missing values in 'label' column; labels must be complete."
                    )
                labels = df["label"].astype(int).tolist()
            urls = None
            if "url" in df.columns:
                urls = df["url"].fillna("").astype(str).tolist()
            return texts, labels, urls
        else:
            raise ValueError("CSV file must contain a 'text' column")

    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if "text" in df.columns:
            texts = df["text"].fillna("").astype(str).tolist()
            labels = None
            if "label" in df.columns:
                if bool(df["label"].isna().any()):
                    raise ValueError(
                        "Parquet file contains missing values in 'label' column; labels must be complete."
                    )
                labels = df["label"].astype(int).tolist()
            urls = None
            if "url" in df.columns:
                urls = df["url"].fillna("").astype(str).tolist()
            return texts, labels, urls
        else:
            raise ValueError("Parquet file must contain a 'text' column")

    elif path.suffix == ".txt":
        # Assume each line is a separate agreement
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()], None, None

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    """
    Example usage of the ExhibitClassifier.
    """
    # Example with synthetic data (replace with your actual data)
    print("M&A Agreement Classifier Example")
    print("=" * 40)

    # Create some example M&A agreement texts (you would replace these with real data)
    ma_agreements = [
        """
        AGREEMENT AND PLAN OF MERGER
        
        This Agreement and Plan of Merger is entered into as of March 15, 2024,
        by and among XYZ Corporation, a Delaware corporation, ABC Acquisition Corp,
        a Delaware corporation and wholly owned subsidiary of XYZ Corporation,
        and Target Company, a Delaware corporation.
        
        RECITALS
        
        WHEREAS, the parties desire to effect a merger of ABC Acquisition Corp
        with and into Target Company;
        
        WHEREAS, the Board of Directors of each party has approved this merger;
        
        NOW, THEREFORE, in consideration of the mutual covenants contained herein,
        the parties agree as follows:
        
        ARTICLE I
        THE MERGER
        
        Section 1.1 The Merger. Upon the terms and subject to the conditions set forth
        in this Agreement, at the Effective Time, ABC Acquisition Corp shall be merged
        with and into Target Company, and Target Company shall be the surviving corporation.
        
        Section 1.2 Merger Consideration. Each share of Target Company common stock
        shall be converted into the right to receive $50.00 in cash.
        """,
        """
        STOCK PURCHASE AGREEMENT
        
        This Stock Purchase Agreement is made on January 10, 2024, between
        Buyer Corp and the stockholders of Seller Inc.
        
        ARTICLE I - PURCHASE AND SALE
        
        Subject to the terms and conditions of this Agreement, Buyer agrees to
        purchase from Sellers, and Sellers agree to sell to Buyer, all of the
        issued and outstanding shares of common stock of the Company for total
        consideration of $100 million.
        
        ARTICLE II - REPRESENTATIONS AND WARRANTIES
        
        The Company and Sellers make various representations and warranties
        regarding the business, financial condition, and legal status.
        
        ARTICLE III - COVENANTS
        
        The parties agree to various covenants including obtaining necessary
        regulatory approvals and HSR Act clearance.
        """,
        """
        ASSET PURCHASE AGREEMENT
        
        This Asset Purchase Agreement is dated as of February 1, 2024, by and between
        Target Corporation and Acquirer LLC.
        
        RECITALS
        
        WHEREAS, Target desires to sell substantially all of its assets to Acquirer;
        WHEREAS, Acquirer desires to acquire such assets;
        
        NOW, THEREFORE, the parties agree:
        
        Section 1. Purchase and Sale of Assets. Target agrees to sell and Acquirer
        agrees to purchase the Purchased Assets for the Purchase Price.
        
        Section 2. Purchase Price. The Purchase Price shall be $75 million in cash.
        """,
        """
        TENDER OFFER AGREEMENT
        
        This Tender Offer Agreement is entered into by BigCorp and SmallCorp
        stockholders. BigCorp hereby commences a tender offer to purchase all
        outstanding shares of SmallCorp common stock at $30.00 per share.
        
        The tender offer is subject to various conditions including minimum
        tender condition, regulatory approvals including HSR Act clearance,
        and other customary closing conditions.
        """,
        """
        MERGER AGREEMENT AND PLAN OF REORGANIZATION
        
        This is a definitive merger agreement between Parent Company and Target Inc.
        The merger will be effected through a reverse triangular merger structure
        where Merger Sub will merge with and into Target, with Target as the
        surviving corporation and becoming a wholly owned subsidiary of Parent.
        
        The merger consideration consists of stock and cash, with an exchange ratio
        of 0.5 shares of Parent stock plus $10 cash for each Target share.
        
        Material adverse change provisions, termination fees, and go-shop provisions
        are included. Fairness opinions have been obtained by both boards.
        """,
        """
        ACQUISITION AGREEMENT
        
        This Acquisition Agreement provides for the acquisition of NewTech Corp
        by Established Corp through a stock-for-stock merger transaction.
        
        The transaction has been approved by both boards of directors and is
        subject to stockholder approval and customary regulatory approvals.
        
        Representations and warranties, covenants, and closing conditions are
        set forth herein. The effective time shall occur upon satisfaction of
        all closing conditions.
        """,
    ]

    non_ma_agreements = [
        """
        EMPLOYMENT AGREEMENT

        This Employment Agreement sets compensation, benefits, and confidentiality
        terms for an executive employee.
        """,
        """
        LICENSE AGREEMENT

        This License Agreement grants software usage rights, support obligations,
        and related service-level commitments.
        """,
    ]

    # Initialize and train classifier
    classifier = ExhibitClassifier()
    training_texts = ma_agreements + non_ma_agreements
    labels = [1] * len(ma_agreements) + [0] * len(non_ma_agreements)
    _ = classifier.fit(training_texts, labels=labels)

    # Test with a clearly M&A-related text
    test_ma_text = """
    MERGER AGREEMENT between Big Corp and Small Corp. The merger consideration
    will be $25 per share. The surviving corporation will be Big Corp.
    Stockholder approval is required. HSR Act filing has been submitted.
    """

    # Test with a non-M&A text
    test_non_ma_text = """
    EMPLOYMENT AGREEMENT between Company and Employee. The employee will
    receive an annual salary of $100,000. This agreement contains standard
    confidentiality and non-compete provisions.
    """

    # Make predictions
    ma_prob = classifier.predict_proba(test_ma_text)
    non_ma_prob = classifier.predict_proba(test_non_ma_text)

    print(f"M&A text probability: {ma_prob:.3f}")
    print(f"Non-M&A text probability: {non_ma_prob:.3f}")

    print(f"M&A text classified as M&A: {classifier.predict(test_ma_text)}")
    print(f"Non-M&A text classified as M&A: {classifier.predict(test_non_ma_text)}")

    # Show feature importance for the M&A text
    print("\nFeature analysis for M&A text:")
    features = classifier.get_feature_importance(test_ma_text)

    # Show top document-level features
    doc_features = {
        k: v
        for k, v in features.items()
        if not k.startswith("hash_word:") and not k.startswith("hash_char:")
    }
    sorted_features = sorted(
        doc_features.items(), key=lambda x: abs(x[1]), reverse=True
    )

    print("Top document features:")
    for name, value in sorted_features[:10]:
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
