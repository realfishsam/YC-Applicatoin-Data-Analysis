import os
from data_loader import prepare_data, get_response_lengths, get_common_questions
from text_analysis import setup_nltk, clean_text, get_word_frequencies, analyze_language_patterns, analyze_sentiment_patterns
from visualization import (
    plot_word_frequencies,
    plot_wordclouds,
    plot_language_patterns,
    plot_response_lengths,
    plot_response_length_by_question,
    plot_sentiment_patterns
)

def ensure_output_dir():
    """Ensure output directory exists."""
    if not os.path.exists('output'):
        os.makedirs('output')

def main():
    """Main function to run the analysis pipeline."""
    print("Setting up NLTK...")
    setup_nltk()
    
    print("Loading and preparing data...")
    data = prepare_data()
    
    # Validate data
    if not data['successful_qna'] and not data['unsuccessful_qna']:
        print("Error: No QnA data found. Please ensure the data has been properly downloaded and processed.")
        return
    
    print(f"Found {len(data['successful_qna'])} successful and {len(data['unsuccessful_qna'])} unsuccessful QnA pairs")
    
    print("Calculating response lengths...")
    successful_lengths = get_response_lengths(data['successful_qna'])
    unsuccessful_lengths = get_response_lengths(data['unsuccessful_qna'])
    
    print("Processing text data...")
    # Combine all answers for each group
    successful_text = ' '.join([qa['answer'] for qa in data['successful_qna']])
    unsuccessful_text = ' '.join([qa['answer'] for qa in data['unsuccessful_qna']])
    
    if not successful_text.strip() and not unsuccessful_text.strip():
        print("Error: No text content found in the QnA pairs")
        return
    
    # Clean text
    successful_clean = clean_text(successful_text)
    unsuccessful_clean = clean_text(unsuccessful_text)
    
    # Get word frequencies
    successful_freq = get_word_frequencies(successful_text)
    unsuccessful_freq = get_word_frequencies(unsuccessful_text)
    
    # Analyze language patterns
    successful_patterns = analyze_language_patterns(successful_text)
    unsuccessful_patterns = analyze_language_patterns(unsuccessful_text)
    
    # Analyze sentiment patterns
    successful_sentiment = analyze_sentiment_patterns(successful_text)
    unsuccessful_sentiment = analyze_sentiment_patterns(unsuccessful_text)
    
    # Get common questions data
    common_questions = get_common_questions(data['all_qna'])
    
    print("Generating visualizations...")
    ensure_output_dir()
    
    # Generate all plots
    plot_word_frequencies(successful_freq, unsuccessful_freq)
    plot_wordclouds(successful_clean, unsuccessful_clean)
    plot_language_patterns(successful_patterns, unsuccessful_patterns)
    plot_response_lengths(successful_lengths, unsuccessful_lengths)
    plot_response_length_by_question(common_questions)
    plot_sentiment_patterns(successful_sentiment, unsuccessful_sentiment)
    
    print("Analysis complete! Results have been saved to the 'output' directory.")

if __name__ == "__main__":
    main() 