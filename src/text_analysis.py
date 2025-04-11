import re
import nltk
import os
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Define the local NLTK data directory
NLTK_DATA_DIR = 'data/nltk_data'

def setup_nltk():
    """Setup NLTK and download required resources."""
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR)
    
    # Download necessary NLTK resources
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK resource {resource}: {e}")
    
    # Add the local directory to NLTK's data path
    nltk.data.path.append(NLTK_DATA_DIR)

def clean_text(text):
    """Clean and preprocess text."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def get_word_frequencies(text):
    """Get word frequencies from text."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters (keep spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    words = text.split()
    
    # Use NLTK stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2 and not word.isdigit()]
    
    # Count frequencies
    return Counter(filtered_words)

def analyze_language_patterns(text):
    """Analyze various language patterns in text."""
    # Check for empty text
    if not text or not text.strip():
        return {
            'firstPersonSingular': 0,
            'firstPersonPlural': 0,
            'future': 0,
            'uncertainty': 0,
            'confidence': 0,
            'passion': 0,
            'numbers': 0,
            'growth': 0
        }

    patterns = {
        'firstPersonSingular': len(re.findall(r'\b(i|me|my|mine)\b', text, re.IGNORECASE)),
        'firstPersonPlural': len(re.findall(r'\b(we|us|our|ours)\b', text, re.IGNORECASE)),
        'future': len(re.findall(r'\b(will|going to|plan to|expect to)\b', text, re.IGNORECASE)),
        'uncertainty': len(re.findall(r'\b(maybe|perhaps|might|could|try|hope|possibly)\b', text, re.IGNORECASE)),
        'confidence': len(re.findall(r'\b(definitely|certainly|absolutely|confident|sure|know|believe)\b', text, re.IGNORECASE)),
        'passion': len(re.findall(r'\b(love|passionate|excited|thrilled|enjoy|care|committed)\b', text, re.IGNORECASE)),
        'numbers': len(re.findall(r'\b\d+(\.\d+)?(k|m|b|million|billion)?\b', text, re.IGNORECASE)),
        'growth': len(re.findall(r'\b(grow|scale|increase|expand|growth)\b', text, re.IGNORECASE))
    }
    
    # Get word count, ensure it's at least 1
    word_count = max(1, len(text.split()))
    normalized_patterns = {k: v / word_count * 1000 for k, v in patterns.items()}  # Per 1000 words
    
    return normalized_patterns

def analyze_sentiment_patterns(text):
    """Analyze sentiment patterns in text."""
    if not text or not text.strip():
        return {
            'polarity': 0,
            'subjectivity': 0,
            'positive_sentences': 0,
            'negative_sentences': 0,
            'neutral_sentences': 0
        }
    
    # Split into sentences
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    
    # Initialize counters
    positive_sentences = 0
    negative_sentences = 0
    neutral_sentences = 0
    total_polarity = 0
    total_subjectivity = 0
    
    # Analyze each sentence
    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        total_polarity += polarity
        total_subjectivity += blob.sentiment.subjectivity
        
        if polarity > 0.1:
            positive_sentences += 1
        elif polarity < -0.1:
            negative_sentences += 1
        else:
            neutral_sentences += 1
    
    # Calculate averages
    avg_polarity = total_polarity / total_sentences if total_sentences > 0 else 0
    avg_subjectivity = total_subjectivity / total_sentences if total_sentences > 0 else 0
    
    return {
        'polarity': avg_polarity,
        'subjectivity': avg_subjectivity,
        'positive_sentences': positive_sentences / total_sentences * 100 if total_sentences > 0 else 0,
        'negative_sentences': negative_sentences / total_sentences * 100 if total_sentences > 0 else 0,
        'neutral_sentences': neutral_sentences / total_sentences * 100 if total_sentences > 0 else 0
    }

def calculate_word_differences(successful_freq, unsuccessful_freq, min_count=3, diff_threshold=0.0005):
    """Calculate differentiating words between successful and unsuccessful applications."""
    # Normalize frequencies
    successful_total = sum(successful_freq.values())
    unsuccessful_total = sum(unsuccessful_freq.values())
    
    successful_norm = {word: count/successful_total for word, count in successful_freq.items()}
    unsuccessful_norm = {word: count/unsuccessful_total for word, count in unsuccessful_freq.items()}
    
    # Find differentiating words
    diff_words = {}
    for word in set(list(successful_norm.keys()) + list(unsuccessful_norm.keys())):
        s_freq_norm = successful_norm.get(word, 0)
        u_freq_norm = unsuccessful_norm.get(word, 0)
        diff = s_freq_norm - u_freq_norm
        
        successful_count = successful_freq.get(word, 0)
        unsuccessful_count = unsuccessful_freq.get(word, 0)
        
        if (successful_count >= min_count or unsuccessful_count >= min_count) and abs(diff) > diff_threshold:
            diff_words[word] = {
                'successful_freq': s_freq_norm,
                'unsuccessful_freq': u_freq_norm,
                'difference': diff,
                'successful_count': successful_count,
                'unsuccessful_count': unsuccessful_count
            }
    
    return diff_words 