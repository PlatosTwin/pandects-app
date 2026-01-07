"""
M&A Agreement Classification for SEC Filings.

This module provides a classifier to determine whether an SEC filing text
represents an M&A agreement. Since we only have positive examples (M&A agreements)
and no labeled negative examples, this uses one-class classification techniques
like One-Class SVM or Isolation Forest.

The classifier extracts document-level features that are characteristic of
M&A agreements and uses anomaly detection to identify non-M&A filings.

Notes:
- With positive-only training data, expect high recall and more false positives; tune `contamination`
  and prediction thresholds based on your tolerance for misses vs. noise.
- Use at least 10 examples; 100+ yields more stable behavior for similarity features.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false

import re
import warnings
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import spmatrix


class ExhibitClassifier:
    """
    One-class classifier for identifying M&A agreements from SEC filing text.
    
    This classifier is designed to work with only positive examples (M&A agreements)
    and uses anomaly detection techniques to identify potential non-M&A filings.
    
    Features extracted include:
    - Document structure indicators (sections, articles, exhibits)
    - Legal language patterns specific to M&A agreements
    - Term frequencies of M&A-specific vocabulary
    - Document length and complexity metrics
    - Semantic similarity to known M&A agreements
    
    Example:
        ```python
        # Train the classifier with M&A agreement texts
        classifier = ExhibitClassifier()
        classifier.fit(ma_agreement_texts)
        
        # Classify a new SEC filing
        probability = classifier.predict_proba("SEC filing text...")
        is_ma_agreement = classifier.predict("SEC filing text...")
        ```
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        max_features: int = 1000,
        char_max_features: int = 2000,
        svd_components: int = 200,
        random_state: int = 42
    ):
        """
        Initialize the M&A agreement classifier.
        
        Args:
            method: Classification method - 'isolation_forest' or 'one_class_svm'
            contamination: Expected proportion of outliers in training data (0.0-0.5)
            max_features: Maximum number of TF-IDF features to use
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.contamination = contamination
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.svd_components = svd_components
        self.random_state = random_state
        
        # Initialize models and transformers
        if method == "isolation_forest":
            self.anomaly_detector = IsolationForest(
                contamination=contamination,  # pyright: ignore[reportArgumentType]
                random_state=random_state,
                n_estimators=100
            )
        elif method == "one_class_svm":
            self.anomaly_detector = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='scale'
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'isolation_forest' or 'one_class_svm'")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        self.char_tfidf_vectorizer = TfidfVectorizer(
            max_features=char_max_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(
            n_components=svd_components,
            random_state=random_state
        )
        
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.training_texts: list[str] = []  # Store for similarity computation
        self.training_tfidf_matrix: spmatrix | None = None
        self.training_scores: np.ndarray | None = None
        self.is_supervised = False
        self.binary_classifier: LogisticRegression | None = None
        
        # M&A-specific vocabulary patterns
        self.ma_keywords = [
            'merger', 'acquisition', 'merger agreement', 'purchase agreement',
            'stock purchase', 'asset purchase', 'merger consideration',
            'surviving corporation', 'constituent corporations', 'merger sub',
            'tender offer', 'exchange ratio', 'merger closing', 'effective time',
            'dissenting shares', 'appraisal rights', 'cash merger', 'stock merger',
            'reverse merger', 'triangular merger', 'short form merger',
            'representations and warranties', 'covenants', 'closing conditions',
            'material adverse change', 'mac', 'material adverse effect', 'mae',
            'termination fee', 'break up fee', 'collar', 'walk away rights',
            'fairness opinion', 'solvency opinion', 'proxy statement',
            'definitive agreement', 'letter of intent', 'due diligence',
            'antitrust clearance', 'hsr act', 'regulatory approval',
            'stockholder approval', 'board recommendation', 'go shop',
            'no shop', 'matching rights', 'superior proposal'
        ]
        
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
        
        # Basic document statistics
        num_chars = len(text)
        words = text.split()
        num_words = len(words)
        num_sentences = len(re.split(r'[.!?]+', text))
        
        # Document structure features
        num_sections = len(re.findall(r'\bsection\s+\d+', text))
        num_articles = len(re.findall(r'\barticle\s+[ivx\d]+', text))
        num_exhibits = len(re.findall(r'\bexhibit\s+[a-z\d]+', text))
        num_schedules = len(re.findall(r'\bschedule\s+[a-z\d\.]+', text))
        
        # Legal document indicators
        has_whereas = 1.0 if 'whereas' in text else 0.0
        has_witnesseth = 1.0 if 'witnesseth' in text or 'w i t n e s s e t h' in text else 0.0
        has_recitals = 1.0 if 'recitals' in text else 0.0
        has_now_therefore = 1.0 if 'now, therefore' in text else 0.0
        
        # M&A-specific language patterns
        ma_keyword_count = sum(1 for keyword in self.ma_keywords if keyword in text)
        ma_keyword_density = ma_keyword_count / num_words if num_words > 0 else 0.0
        
        # Specific M&A terms
        has_merger = 1.0 if 'merger' in text else 0.0
        has_acquisition = 1.0 if 'acquisition' in text or 'acquire' in text else 0.0
        has_purchase = 1.0 if 'purchase' in text else 0.0
        has_consideration = 1.0 if 'consideration' in text else 0.0
        has_closing = 1.0 if 'closing' in text else 0.0
        has_effective_time = 1.0 if 'effective time' in text else 0.0
        has_surviving = 1.0 if 'surviving' in text else 0.0
        
        # Corporate structure terms
        has_subsidiary = 1.0 if 'subsidiary' in text else 0.0
        has_stockholder = 1.0 if 'stockholder' in text or 'shareholder' in text else 0.0
        has_board = 1.0 if 'board of directors' in text else 0.0
        
        # Regulatory and approval terms
        has_hsr = 1.0 if 'hsr' in text or 'hart-scott-rodino' in text else 0.0
        has_antitrust = 1.0 if 'antitrust' in text else 0.0
        has_regulatory = 1.0 if 'regulatory approval' in text else 0.0
        has_proxy = 1.0 if 'proxy' in text else 0.0
        
        # Financial terms
        has_price = 1.0 if any(term in text for term in ['price per share', 'exchange ratio', 'cash consideration']) else 0.0
        has_fairness = 1.0 if 'fairness opinion' in text else 0.0
        has_valuation = 1.0 if 'valuation' in text else 0.0
        
        # Document complexity
        avg_words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0.0
        avg_chars_per_word = num_chars / num_words if num_words > 0 else 0.0
        
        # Legal boilerplate density
        boilerplate_terms = ['hereto', 'herein', 'hereby', 'thereof', 'wherein', 'heretofore', 'hereinafter']
        boilerplate_count = sum(text.count(term) for term in boilerplate_terms)
        boilerplate_density = boilerplate_count / num_words if num_words > 0 else 0.0
        
        # Punctuation analysis (legal documents have specific patterns)
        semicolon_density = text.count(';') / num_chars if num_chars > 0 else 0.0
        paren_density = text.count('(') / num_chars if num_chars > 0 else 0.0
        
        # Compile feature vector
        features = [
            num_words,
            num_chars,
            num_sentences,
            num_sections,
            num_articles, 
            num_exhibits,
            num_schedules,
            has_whereas,
            has_witnesseth,
            has_recitals,
            has_now_therefore,
            ma_keyword_count,
            ma_keyword_density,
            has_merger,
            has_acquisition,
            has_purchase,
            has_consideration,
            has_closing,
            has_effective_time,
            has_surviving,
            has_subsidiary,
            has_stockholder,
            has_board,
            has_hsr,
            has_antitrust,
            has_regulatory,
            has_proxy,
            has_price,
            has_fairness,
            has_valuation,
            avg_words_per_sentence,
            avg_chars_per_word,
            boilerplate_density,
            semicolon_density,
            paren_density
        ]
        
        return np.array(features, dtype=float)
    
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
            self.training_tfidf_matrix = cast(
                spmatrix, self.tfidf_vectorizer.transform(self.training_texts)
            )

        query_vector = self.tfidf_vectorizer.transform([text])
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
            self.training_tfidf_matrix = cast(
                spmatrix, self.tfidf_vectorizer.transform(self.training_texts)
            )

        query_matrix = self.tfidf_vectorizer.transform(texts)
        training_vectors = self.training_tfidf_matrix

        similarities = cosine_similarity(query_matrix, training_vectors)
        max_similarity = np.max(similarities, axis=1)
        mean_similarity = np.mean(similarities, axis=1)
        median_similarity = np.median(similarities, axis=1)
        return np.vstack([max_similarity, mean_similarity, median_similarity]).T

    def _build_text_matrix(self, texts: list[str]) -> spmatrix:
        word_matrix = cast(spmatrix, self.tfidf_vectorizer.transform(texts))
        char_matrix = cast(spmatrix, self.char_tfidf_vectorizer.transform(texts))
        return cast(spmatrix, sparse_hstack([word_matrix, char_matrix]))

    @staticmethod
    def _compute_training_similarity_features(tfidf_matrix: spmatrix) -> np.ndarray:
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
            labels: Optional list of 0/1 labels for supervised training
            
        Returns:
            Self for method chaining
        """
        if len(texts) < 10:
            warnings.warn(
                f"Training with only {len(texts)} examples. Consider using more training data for better performance."
            )
        
        print(f"Training M&A classifier with {len(texts)} agreements...")
        
        # Store training texts for similarity computation
        self.training_texts = texts.copy()
        
        # Fit TF-IDF vectorizers
        _ = self.tfidf_vectorizer.fit(texts)
        _ = self.char_tfidf_vectorizer.fit(texts)
        
        # Extract document-level features
        doc_features = np.array([self._extract_document_features(text) for text in texts])
        
        # Extract TF-IDF features (use smaller subset for efficiency)
        tfidf_matrix = self._build_text_matrix(texts)
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        self.training_tfidf_matrix = cast(
            spmatrix, self.tfidf_vectorizer.transform(texts)
        )
        
        similarity_features = self._compute_training_similarity_features(tfidf_matrix)
        
        # Combine all features
        all_features = np.hstack([doc_features, tfidf_reduced, similarity_features])
        
        # Scale features
        _ = self.feature_scaler.fit(all_features)
        scaled_features = self.feature_scaler.transform(all_features)

        self.is_supervised = bool(labels) and any(label == 0 for label in labels)
        if self.is_supervised:
            y = np.array(labels, dtype=int)
            self.binary_classifier = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
            _ = self.binary_classifier.fit(scaled_features, y)
            self.training_scores = self.binary_classifier.predict_proba(
                scaled_features
            )[:, 1]
        else:
            # Train anomaly detector
            _ = self.anomaly_detector.fit(scaled_features)
            self.training_scores = self.anomaly_detector.decision_function(
                scaled_features
            )
        
        self.is_fitted = True
        print(f"Classifier trained successfully using {self.method}")
        
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
        tfidf_features = self.svd.transform(tfidf_matrix)[0]
        similarity_features = self._compute_similarity_features(text)
        
        # Combine features
        all_features = np.hstack([doc_features, tfidf_features, similarity_features])
        scaled_features = self.feature_scaler.transform([all_features])
        
        # Get anomaly score
        if self.is_supervised and self.binary_classifier is not None:
            probability = self.binary_classifier.predict_proba(scaled_features)[0, 1]
            return float(np.clip(probability, 0.0, 1.0))

        score = float(self.anomaly_detector.decision_function(scaled_features)[0])
        if self.training_scores is None:
            return float(np.clip(1 / (1 + np.exp(-score)), 0.0, 1.0))

        sorted_scores = np.sort(self.training_scores)
        rank = np.searchsorted(sorted_scores, score, side="left")
        probability = rank / max(len(sorted_scores) - 1, 1)
        return float(np.clip(probability, 0.0, 1.0))

    def predict_proba_batch(self, texts: list[str]) -> list[float]:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before making predictions")
        if not texts:
            return []

        doc_features = np.array([self._extract_document_features(text) for text in texts])
        tfidf_matrix = self._build_text_matrix(texts)
        tfidf_features = self.svd.transform(tfidf_matrix)
        similarity_features = self._compute_similarity_features_batch(texts)

        all_features = np.hstack([doc_features, tfidf_features, similarity_features])
        scaled_features = self.feature_scaler.transform(all_features)

        if self.is_supervised and self.binary_classifier is not None:
            probabilities = self.binary_classifier.predict_proba(scaled_features)[:, 1]
            return [float(p) for p in np.clip(probabilities, 0.0, 1.0)]

        scores = self.anomaly_detector.decision_function(scaled_features)
        if self.training_scores is None:
            probabilities = 1 / (1 + np.exp(-scores))
            return [float(p) for p in np.clip(probabilities, 0.0, 1.0)]

        sorted_scores = np.sort(self.training_scores)
        ranks = np.searchsorted(sorted_scores, scores, side="left")
        probabilities = ranks / max(len(sorted_scores) - 1, 1)
        return [float(p) for p in np.clip(probabilities, 0.0, 1.0)]

    def predict_batch(self, texts: list[str], threshold: float = 0.5) -> list[bool]:
        probabilities = self.predict_proba_batch(texts)
        return [prob >= threshold for prob in probabilities]
    
    def predict(self, text: str, threshold: float = 0.5) -> bool:
        """
        Predict whether the text is an M&A agreement.
        
        Args:
            text: SEC filing text content
            threshold: Probability threshold for classification
            
        Returns:
            True if predicted to be M&A agreement, False otherwise
        """
        probability = self.predict_proba(text)
        return probability >= threshold
    
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
            'anomaly_detector': self.anomaly_detector,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'char_tfidf_vectorizer': self.char_tfidf_vectorizer,
            'svd': self.svd,
            'feature_scaler': self.feature_scaler,
            'training_texts': self.training_texts,
            'training_tfidf_matrix': self.training_tfidf_matrix,
            'training_scores': self.training_scores,
            'is_supervised': self.is_supervised,
            'binary_classifier': self.binary_classifier,
            'method': self.method,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'char_max_features': self.char_max_features,
            'svd_components': self.svd_components,
            'random_state': self.random_state,
            'ma_keywords': self.ma_keywords
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
            method=model_data['method'],
            contamination=model_data['contamination'],
            max_features=model_data['max_features'],
            random_state=model_data['random_state']
        )
        
        # Restore trained components
        classifier.anomaly_detector = model_data['anomaly_detector']
        classifier.tfidf_vectorizer = model_data['tfidf_vectorizer']
        classifier.char_tfidf_vectorizer = model_data['char_tfidf_vectorizer']
        classifier.svd = model_data['svd']
        classifier.feature_scaler = model_data['feature_scaler']
        classifier.training_texts = model_data['training_texts']
        classifier.training_tfidf_matrix = model_data.get('training_tfidf_matrix')
        classifier.training_scores = model_data.get('training_scores')
        classifier.is_supervised = model_data.get('is_supervised', False)
        classifier.binary_classifier = model_data.get('binary_classifier')
        classifier.ma_keywords = model_data['ma_keywords']
        classifier.is_fitted = True
        
        print(f"Classifier loaded from {path}")
        return classifier
    
    def get_feature_importance(self, text: str) -> dict[str, float]:
        """
        Get feature importance for a given text (for interpretation).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping feature names to their values
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before feature analysis")
        
        # Extract features
        doc_features = self._extract_document_features(text)
        tfidf_matrix = cast(Any, self.tfidf_vectorizer.transform([text]))
        tfidf_features = cast(np.ndarray, tfidf_matrix.toarray())[0]
        similarity_features = self._compute_similarity_features(text)
        
        # Feature names
        doc_feature_names = [
            'num_words', 'num_chars', 'num_sentences', 'num_sections', 'num_articles',
            'num_exhibits', 'num_schedules', 'has_whereas', 'has_witnesseth', 
            'has_recitals', 'has_now_therefore', 'ma_keyword_count', 'ma_keyword_density',
            'has_merger', 'has_acquisition', 'has_purchase', 'has_consideration',
            'has_closing', 'has_effective_time', 'has_surviving', 'has_subsidiary',
            'has_stockholder', 'has_board', 'has_hsr', 'has_antitrust', 'has_regulatory',
            'has_proxy', 'has_price', 'has_fairness', 'has_valuation',
            'avg_words_per_sentence', 'avg_chars_per_word', 'boilerplate_density',
            'semicolon_density', 'paren_density'
        ]
        
        tfidf_feature_names = [
            f"tfidf_{word}" for word in self.tfidf_vectorizer.get_feature_names_out()
        ]
        similarity_feature_names = ['max_similarity', 'mean_similarity', 'median_similarity']
        
        all_feature_names = doc_feature_names + tfidf_feature_names + similarity_feature_names
        all_feature_values = np.hstack([doc_features, tfidf_features, similarity_features])
        
        return dict(zip(all_feature_names, all_feature_values))


def load_training_data(
    data_path: str,
) -> tuple[list[str], list[int] | None]:
    """
    Load M&A agreement texts from a data file.
    
    This is a helper function to load training data. The format depends on
    how your M&A agreement data is stored.
    
    Args:
        data_path: Path to the training data file
        
    Returns:
        Tuple of text list and optional label list
    """
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        # Assume there's a 'text' column with the agreement content
        if 'text' in df.columns:
            texts = df['text'].fillna('').astype(str).tolist()
            labels = None
            if 'label' in df.columns:
                labels = df['label'].fillna(1).astype(int).tolist()
            return texts, labels
        else:
            raise ValueError("CSV file must contain a 'text' column")
    
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if 'text' in df.columns:
            texts = df['text'].fillna('').astype(str).tolist()
            labels = None
            if 'label' in df.columns:
                labels = df['label'].fillna(1).astype(int).tolist()
            return texts, labels
        else:
            raise ValueError("Parquet file must contain a 'text' column")
    
    elif path.suffix == ".txt":
        # Assume each line is a separate agreement
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()], None
    
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
        """
    ]
    
    # Initialize and train classifier
    classifier = ExhibitClassifier(method="isolation_forest", contamination=0.1)
    _ = classifier.fit(ma_agreements)
    
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
    doc_features = {k: v for k, v in features.items() if not k.startswith('tfidf_')}
    sorted_features = sorted(doc_features.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Top document features:")
    for name, value in sorted_features[:10]:
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
