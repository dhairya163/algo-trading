"""
Sentiment analysis engine using multiple NLP approaches.

Supports:
- FinBERT (Financial domain BERT model)
- VADER (Rule-based sentiment)
- TextBlob (Pattern-based sentiment)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from src.data.models import NewsArticle, SentimentLabel


class AnalysisMethod(str, Enum):
    """Sentiment analysis methods."""

    FINBERT = "finbert"
    VADER = "vader"
    TEXTBLOB = "textblob"
    ENSEMBLE = "ensemble"


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    score: float  # -1 to 1
    label: SentimentLabel
    confidence: float  # 0 to 1
    method: AnalysisMethod
    positive_prob: Optional[float] = None
    negative_prob: Optional[float] = None
    neutral_prob: Optional[float] = None


class SentimentAnalyzer:
    """
    Multi-method sentiment analyzer for financial text.

    Uses FinBERT for best accuracy, with fallback to VADER/TextBlob.
    """

    def __init__(self, default_method: AnalysisMethod = AnalysisMethod.ENSEMBLE):
        """
        Initialize sentiment analyzer.

        Args:
            default_method: Default analysis method to use
        """
        self.default_method = default_method
        self._finbert_model = None
        self._finbert_tokenizer = None
        self._vader_analyzer = None
        self._initialized = False

    def _init_finbert(self):
        """Initialize FinBERT model (lazy loading)."""
        if self._finbert_model is not None:
            return True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "ProsusAI/finbert"
            logger.info(f"Loading FinBERT model: {model_name}")

            self._finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self._finbert_model = self._finbert_model.cuda()
                logger.info("FinBERT model loaded on GPU")
            else:
                logger.info("FinBERT model loaded on CPU")

            return True

        except ImportError:
            logger.warning("transformers/torch not installed, FinBERT unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return False

    def _init_vader(self):
        """Initialize VADER analyzer."""
        if self._vader_analyzer is not None:
            return True

        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            # Download VADER lexicon if needed
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)

            self._vader_analyzer = SentimentIntensityAnalyzer()

            # Add financial domain words to VADER lexicon
            financial_lexicon = {
                "bullish": 2.5,
                "bearish": -2.5,
                "upgrade": 2.0,
                "downgrade": -2.0,
                "outperform": 2.0,
                "underperform": -2.0,
                "beat": 1.5,
                "miss": -1.5,
                "surge": 2.0,
                "plunge": -2.0,
                "rally": 2.0,
                "crash": -2.5,
                "boom": 2.0,
                "bust": -2.0,
                "profit": 1.5,
                "loss": -1.5,
                "growth": 1.5,
                "decline": -1.5,
                "record": 1.0,
                "bankruptcy": -3.0,
                "default": -2.5,
                "dividend": 1.0,
                "acquisition": 0.5,
                "merger": 0.5,
                "layoff": -1.5,
                "restructuring": -0.5,
            }

            self._vader_analyzer.lexicon.update(financial_lexicon)
            logger.info("VADER analyzer initialized with financial lexicon")
            return True

        except ImportError:
            logger.warning("nltk not installed, VADER unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")
            return False

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label."""
        if score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLabel.NEGATIVE
        elif score <= 0.2:
            return SentimentLabel.NEUTRAL
        elif score <= 0.6:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.VERY_POSITIVE

    def analyze_finbert(self, text: str) -> Optional[SentimentResult]:
        """
        Analyze sentiment using FinBERT.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult or None if FinBERT unavailable
        """
        if not self._init_finbert():
            return None

        try:
            import torch

            # Tokenize and prepare input
            inputs = self._finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Move to GPU if model is on GPU
            if next(self._finbert_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self._finbert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT labels: positive, negative, neutral
            probs = probs.cpu().numpy()[0]
            positive_prob = float(probs[0])
            negative_prob = float(probs[1])
            neutral_prob = float(probs[2])

            # Calculate score (-1 to 1)
            score = positive_prob - negative_prob

            # Confidence is max probability
            confidence = max(positive_prob, negative_prob, neutral_prob)

            return SentimentResult(
                score=score,
                label=self._score_to_label(score),
                confidence=confidence,
                method=AnalysisMethod.FINBERT,
                positive_prob=positive_prob,
                negative_prob=negative_prob,
                neutral_prob=neutral_prob,
            )

        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return None

    def analyze_vader(self, text: str) -> Optional[SentimentResult]:
        """
        Analyze sentiment using VADER.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult or None if VADER unavailable
        """
        if not self._init_vader():
            return None

        try:
            scores = self._vader_analyzer.polarity_scores(text)

            # VADER compound score is already -1 to 1
            score = scores["compound"]

            # Calculate confidence based on how polarized the sentiment is
            confidence = abs(score) * 0.7 + 0.3  # Base confidence of 0.3

            return SentimentResult(
                score=score,
                label=self._score_to_label(score),
                confidence=min(confidence, 1.0),
                method=AnalysisMethod.VADER,
                positive_prob=scores["pos"],
                negative_prob=scores["neg"],
                neutral_prob=scores["neu"],
            )

        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return None

    def analyze_textblob(self, text: str) -> Optional[SentimentResult]:
        """
        Analyze sentiment using TextBlob.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult or None if TextBlob unavailable
        """
        try:
            from textblob import TextBlob

            blob = TextBlob(text)

            # TextBlob polarity is -1 to 1
            score = blob.sentiment.polarity

            # Subjectivity can indicate confidence (more subjective = more confident sentiment)
            subjectivity = blob.sentiment.subjectivity
            confidence = 0.3 + (subjectivity * 0.5) + (abs(score) * 0.2)

            return SentimentResult(
                score=score,
                label=self._score_to_label(score),
                confidence=min(confidence, 1.0),
                method=AnalysisMethod.TEXTBLOB,
            )

        except ImportError:
            logger.warning("textblob not installed")
            return None
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return None

    def analyze_ensemble(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using ensemble of all available methods.

        Weights results by confidence and method reliability.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with weighted average
        """
        results = []
        weights = []

        # FinBERT (highest weight - domain specific)
        finbert_result = self.analyze_finbert(text)
        if finbert_result:
            results.append(finbert_result)
            weights.append(0.5)  # 50% weight

        # VADER (medium weight - good for social/news text)
        vader_result = self.analyze_vader(text)
        if vader_result:
            results.append(vader_result)
            weights.append(0.35)  # 35% weight

        # TextBlob (lowest weight - general purpose)
        textblob_result = self.analyze_textblob(text)
        if textblob_result:
            results.append(textblob_result)
            weights.append(0.15)  # 15% weight

        if not results:
            # Return neutral if all methods failed
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                method=AnalysisMethod.ENSEMBLE,
            )

        # Normalize weights
        total_weight = sum(weights[: len(results)])
        weights = [w / total_weight for w in weights[: len(results)]]

        # Calculate weighted average
        weighted_score = sum(r.score * w for r, w in zip(results, weights))
        weighted_confidence = sum(r.confidence * w for r, w in zip(results, weights))

        # Average probabilities if available
        pos_probs = [r.positive_prob for r in results if r.positive_prob is not None]
        neg_probs = [r.negative_prob for r in results if r.negative_prob is not None]
        neu_probs = [r.neutral_prob for r in results if r.neutral_prob is not None]

        return SentimentResult(
            score=weighted_score,
            label=self._score_to_label(weighted_score),
            confidence=weighted_confidence,
            method=AnalysisMethod.ENSEMBLE,
            positive_prob=sum(pos_probs) / len(pos_probs) if pos_probs else None,
            negative_prob=sum(neg_probs) / len(neg_probs) if neg_probs else None,
            neutral_prob=sum(neu_probs) / len(neu_probs) if neu_probs else None,
        )

    def analyze(
        self,
        text: str,
        method: Optional[AnalysisMethod] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment using specified or default method.

        Args:
            text: Text to analyze
            method: Analysis method (default: ensemble)

        Returns:
            SentimentResult
        """
        method = method or self.default_method

        if method == AnalysisMethod.FINBERT:
            result = self.analyze_finbert(text)
        elif method == AnalysisMethod.VADER:
            result = self.analyze_vader(text)
        elif method == AnalysisMethod.TEXTBLOB:
            result = self.analyze_textblob(text)
        else:  # ENSEMBLE
            result = self.analyze_ensemble(text)

        if result is None:
            # Fallback to ensemble if specific method failed
            result = self.analyze_ensemble(text)

        return result

    def analyze_article(self, article: NewsArticle) -> SentimentResult:
        """
        Analyze sentiment of a news article.

        Combines title and summary/content for analysis.

        Args:
            article: NewsArticle to analyze

        Returns:
            SentimentResult
        """
        # Combine title and content with emphasis on title
        text_parts = [article.title]

        if article.summary:
            text_parts.append(article.summary)
        elif article.content:
            # Use first 500 chars of content
            text_parts.append(article.content[:500])

        combined_text = " ".join(text_parts)

        return self.analyze(combined_text)

    def analyze_batch(
        self,
        texts: list[str],
        method: Optional[AnalysisMethod] = None,
    ) -> list[SentimentResult]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze
            method: Analysis method

        Returns:
            List of SentimentResults
        """
        return [self.analyze(text, method) for text in texts]
