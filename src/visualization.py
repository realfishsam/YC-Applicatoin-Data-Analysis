import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

def plot_word_frequencies(successful_freq, unsuccessful_freq, output_file='output/top_words_comparison.png'):
    """Plot top words comparison between successful and unsuccessful applications."""
    if not successful_freq and not unsuccessful_freq:
        print("Warning: No words to plot in word frequencies comparison")
        return
        
    plt.figure(figsize=(12, 10))
    
    # Top words in successful applications
    successful_top_words = dict(sorted(successful_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    plt.subplot(2, 1, 1)
    if successful_top_words:
        sns.barplot(x=list(successful_top_words.values()), y=list(successful_top_words.keys()), palette='viridis')
    plt.title('Top 20 Words in Successful Applications')
    plt.xlabel('Frequency')
    
    # Top words in unsuccessful applications
    unsuccessful_top_words = dict(sorted(unsuccessful_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    plt.subplot(2, 1, 2)
    if unsuccessful_top_words:
        sns.barplot(x=list(unsuccessful_top_words.values()), y=list(unsuccessful_top_words.keys()), palette='viridis')
    plt.title('Top 20 Words in Unsuccessful Applications')
    plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_wordclouds(successful_text, unsuccessful_text, output_file='output/wordcloud_comparison.png'):
    """Generate and plot word clouds for successful and unsuccessful applications."""
    if not successful_text.strip() and not unsuccessful_text.strip():
        print("Warning: No text to generate word clouds")
        return
        
    plt.figure(figsize=(18, 8))
    
    # Word cloud for successful applications
    plt.subplot(1, 2, 1)
    if successful_text.strip():
        successful_wordcloud = WordCloud(width=800, height=400, background_color='white',
                                      max_words=100, colormap='viridis').generate(successful_text)
        plt.imshow(successful_wordcloud, interpolation='bilinear')
    plt.title('Words in Successful Applications', fontsize=20)
    plt.axis('off')
    
    # Word cloud for unsuccessful applications
    plt.subplot(1, 2, 2)
    if unsuccessful_text.strip():
        unsuccessful_wordcloud = WordCloud(width=800, height=400, background_color='white',
                                        max_words=100, colormap='viridis').generate(unsuccessful_text)
        plt.imshow(unsuccessful_wordcloud, interpolation='bilinear')
    plt.title('Words in Unsuccessful Applications', fontsize=20)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_language_patterns(successful_patterns, unsuccessful_patterns, output_file='output/language_patterns.png'):
    """Plot language pattern comparison between successful and unsuccessful applications."""
    if not successful_patterns and not unsuccessful_patterns:
        print("Warning: No language patterns to plot")
        return
        
    plt.figure(figsize=(14, 8))
    patterns = list(successful_patterns.keys())
    successful_values = [successful_patterns[p] for p in patterns]
    unsuccessful_values = [unsuccessful_patterns[p] for p in patterns]
    
    x = range(len(patterns))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar([i - width/2 for i in x], successful_values, width, label='Successful', color='#4CAF50')
    rects2 = ax.bar([i + width/2 for i in x], unsuccessful_values, width, label='Unsuccessful', color='#F44336')
    
    ax.set_ylabel('Occurrences per 1000 words')
    ax.set_title('Language Patterns in Applications')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_response_lengths(successful_lengths, unsuccessful_lengths, output_file='output/response_length_comparison.png'):
    """Plot response length comparison between successful and unsuccessful applications."""
    if not successful_lengths and not unsuccessful_lengths:
        print("Warning: No response lengths to plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    successful_avg = sum(d['length'] for d in successful_lengths) / max(len(successful_lengths), 1)
    unsuccessful_avg = sum(d['length'] for d in unsuccessful_lengths) / max(len(unsuccessful_lengths), 1)
    
    plt.bar(['Successful', 'Unsuccessful'], [successful_avg, unsuccessful_avg], color=['#4CAF50', '#F44336'])
    plt.title('Average Response Length by Application Status')
    plt.ylabel('Average Number of Words')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_response_length_by_question(common_questions, output_file='output/response_length_by_question.png'):
    """Plot response length comparison by question."""
    if not common_questions:
        print("Warning: No common questions to plot")
        return

    # Find questions with both successful and unsuccessful responses
    questions_with_both = {}
    for question, data in common_questions.items():
        if data['successful'] and data['unsuccessful']:
            successful_avg = sum(data['successful']) / len(data['successful'])
            unsuccessful_avg = sum(data['unsuccessful']) / len(data['unsuccessful'])
            difference = unsuccessful_avg - successful_avg
            
            questions_with_both[question] = {
                'successful_avg': successful_avg,
                'unsuccessful_avg': unsuccessful_avg,
                'difference': difference
            }

    # Sort by absolute difference
    sorted_questions = sorted(questions_with_both.items(), key=lambda x: abs(x[1]['difference']), reverse=True)

    # Visualize top differentiating questions
    top_n = min(10, len(sorted_questions))
    questions = [q[:40] + '...' if len(q) > 40 else q for q, _ in sorted_questions[:top_n]]
    successful_avgs = [data['successful_avg'] for _, data in sorted_questions[:top_n]]
    unsuccessful_avgs = [data['unsuccessful_avg'] for _, data in sorted_questions[:top_n]]

    plt.figure(figsize=(14, 10))
    x = np.arange(len(questions))  # Use numpy.arange instead of range
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 10))
    rects1 = ax.barh(x - width/2, successful_avgs, width, label='Successful', color='#4CAF50')
    rects2 = ax.barh(x + width/2, unsuccessful_avgs, width, label='Unsuccessful', color='#F44336')

    ax.set_xlabel('Average Response Length (words)')
    ax.set_title('Response Length Comparison by Question')
    ax.set_yticks(x)
    ax.set_yticklabels(questions)
    ax.legend()

    # Add value labels
    def autolabel_h(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate(f'{int(width)}',
                        xy=(width, rect.get_y() + rect.get_height()/2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    autolabel_h(rects1)
    autolabel_h(rects2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_sentiment_patterns(successful_sentiment, unsuccessful_sentiment, output_file='output/sentiment_patterns.png'):
    """Plot sentiment pattern comparison between successful and unsuccessful applications."""
    if not successful_sentiment and not unsuccessful_sentiment:
        print("Warning: No sentiment patterns to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot polarity and subjectivity
    metrics = ['polarity', 'subjectivity']
    successful_values = [successful_sentiment[m] for m in metrics]
    unsuccessful_values = [unsuccessful_sentiment[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, successful_values, width, label='Successful', color='#4CAF50')
    ax1.bar(x + width/2, unsuccessful_values, width, label='Unsuccessful', color='#F44336')
    ax1.set_ylabel('Score')
    ax1.set_title('Sentiment Analysis: Polarity and Subjectivity')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Plot sentence distribution
    metrics = ['positive_sentences', 'negative_sentences', 'neutral_sentences']
    successful_values = [successful_sentiment[m] for m in metrics]
    unsuccessful_values = [unsuccessful_sentiment[m] for m in metrics]
    
    x = np.arange(len(metrics))
    
    ax2.bar(x - width/2, successful_values, width, label='Successful', color='#4CAF50')
    ax2.bar(x + width/2, unsuccessful_values, width, label='Unsuccessful', color='#F44336')
    ax2.set_ylabel('Percentage of Sentences')
    ax2.set_title('Sentence Sentiment Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close() 