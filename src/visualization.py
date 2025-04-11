import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd

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

def plot_normalized_word_frequencies(successful_freq, unsuccessful_freq, output_file='output/normalized_word_comparison.png'):
    """Plot normalized word frequencies comparison between successful and unsuccessful applications."""
    if not successful_freq and not unsuccessful_freq:
        print("Warning: No words to plot in normalized word frequencies comparison")
        return
        
    # Normalize by total word count in each category
    successful_total = sum(successful_freq.values())
    unsuccessful_total = sum(unsuccessful_freq.values())
    
    # Calculate normalized frequencies (per 1000 words)
    successful_norm = {word: (count/successful_total) * 1000 for word, count in successful_freq.items()}
    unsuccessful_norm = {word: (count/unsuccessful_total) * 1000 for word, count in unsuccessful_freq.items()}
    
    # Get top words by normalized frequency
    successful_top_words = dict(sorted(successful_norm.items(), key=lambda x: x[1], reverse=True)[:20])
    unsuccessful_top_words = dict(sorted(unsuccessful_norm.items(), key=lambda x: x[1], reverse=True)[:20])
    
    plt.figure(figsize=(12, 10))
    
    # Top words in successful applications
    plt.subplot(2, 1, 1)
    if successful_top_words:
        sns.barplot(x=list(successful_top_words.values()), y=list(successful_top_words.keys()), palette='viridis')
    plt.title('Top 20 Words in Successful Applications (Normalized per 1000 Words)')
    plt.xlabel('Frequency per 1000 Words')
    
    # Top words in unsuccessful applications
    plt.subplot(2, 1, 2)
    if unsuccessful_top_words:
        sns.barplot(x=list(unsuccessful_top_words.values()), y=list(unsuccessful_top_words.keys()), palette='viridis')
    plt.title('Top 20 Words in Unsuccessful Applications (Normalized per 1000 Words)')
    plt.xlabel('Frequency per 1000 Words')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_differentiating_words(successful_freq, unsuccessful_freq, output_file='output/differentiating_words.png'):
    """Plot words that differentiate successful from unsuccessful applications."""
    if not successful_freq and not unsuccessful_freq:
        print("Warning: No words to plot for differentiating words")
        return
    
    # Get total word counts
    successful_total = sum(successful_freq.values())
    unsuccessful_total = sum(unsuccessful_freq.values())
    
    # Normalize frequencies
    successful_norm = {word: count/successful_total for word, count in successful_freq.items()}
    unsuccessful_norm = {word: count/unsuccessful_total for word, count in unsuccessful_freq.items()}
    
    # Calculate difference for each word
    diff_words = {}
    for word in set(list(successful_norm.keys()) + list(unsuccessful_norm.keys())):
        s_freq = successful_norm.get(word, 0)
        u_freq = unsuccessful_norm.get(word, 0)
        diff = s_freq - u_freq
        
        # Only consider words that appear at least a few times
        successful_count = successful_freq.get(word, 0)
        unsuccessful_count = unsuccessful_freq.get(word, 0)
        
        if (successful_count >= 5 or unsuccessful_count >= 5) and abs(diff) > 0.0005:
            diff_words[word] = diff
    
    # Get top differentiating words in each direction
    more_in_successful = dict(sorted([(w, d) for w, d in diff_words.items() if d > 0], 
                                     key=lambda x: x[1], reverse=True)[:10])
    more_in_unsuccessful = dict(sorted([(w, d) for w, d in diff_words.items() if d < 0], 
                                       key=lambda x: x[1])[:10])
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Words more common in successful applications
    ax1.barh(list(more_in_successful.keys()), 
             [d * 1000 for d in more_in_successful.values()], 
             color='#4CAF50')
    ax1.set_title('Words More Common in Successful Applications')
    ax1.set_xlabel('Difference in Frequency (per 1000 words)')
    
    # Words more common in unsuccessful applications
    ax2.barh(list(more_in_unsuccessful.keys()), 
             [abs(d * 1000) for d in more_in_unsuccessful.values()], 
             color='#F44336')
    ax2.set_title('Words More Common in Unsuccessful Applications')
    ax2.set_xlabel('Difference in Frequency (per 1000 words)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_normalized_response_density(successful_qna, unsuccessful_qna, output_file='output/normalized_response_density.png'):
    """Plot the normalized response density (words per answer) for each group."""
    if not successful_qna and not unsuccessful_qna:
        print("Warning: No QnA pairs to plot for normalized response density")
        return
    
    # Group questions by type and calculate average length for each
    def get_question_avg_lengths(qna_pairs):
        question_lengths = {}
        for qa in qna_pairs:
            question = qa['question']
            length = len(qa['answer'].split())
            
            if question not in question_lengths:
                question_lengths[question] = []
            question_lengths[question].append(length)
        
        return {q: sum(lengths)/len(lengths) for q, lengths in question_lengths.items()}
    
    successful_q_avgs = get_question_avg_lengths(successful_qna)
    unsuccessful_q_avgs = get_question_avg_lengths(unsuccessful_qna)
    
    # Get all unique questions
    all_questions = set(list(successful_q_avgs.keys()) + list(unsuccessful_q_avgs.keys()))
    
    # Create normalized response length data
    data = []
    for question in all_questions:
        if question in successful_q_avgs:
            data.append({
                'question': question,
                'group': 'Successful',
                'avg_length': successful_q_avgs[question]
            })
        if question in unsuccessful_q_avgs:
            data.append({
                'question': question,
                'group': 'Unsuccessful',
                'avg_length': unsuccessful_q_avgs[question]
            })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data)
    
    # Create violin plot of response lengths
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='group', y='avg_length', data=df, palette={'Successful': '#4CAF50', 'Unsuccessful': '#F44336'})
    plt.title('Distribution of Response Lengths (Normalized by Question)')
    plt.xlabel('Application Status')
    plt.ylabel('Average Words per Answer')
    
    # Add a box plot inside the violin plot for more detail
    sns.boxplot(x='group', y='avg_length', data=df, palette={'Successful': '#4CAF50', 'Unsuccessful': '#F44336'}, 
                width=0.3, boxprops={'alpha': 0.6})
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_normalized_language_patterns(successful_patterns, unsuccessful_patterns, output_file='output/normalized_language_patterns.png'):
    """Plot normalized language pattern comparison showing relative differences."""
    if not successful_patterns and not unsuccessful_patterns:
        print("Warning: No language patterns to plot for normalization")
        return
        
    # Calculate percentage differences
    patterns = list(successful_patterns.keys())
    differences = []
    labels = []
    colors = []
    
    for pattern in patterns:
        s_value = successful_patterns.get(pattern, 0)
        u_value = unsuccessful_patterns.get(pattern, 0)
        
        # Calculate percentage difference relative to unsuccessful
        if u_value != 0:
            diff_percent = ((s_value - u_value) / u_value) * 100
        else:
            diff_percent = 100 if s_value > 0 else 0
            
        differences.append(diff_percent)
        labels.append(pattern)
        colors.append('#4CAF50' if diff_percent >= 0 else '#F44336')
    
    # Create plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    
    # Sort by absolute percentage difference
    sorted_indices = np.argsort(np.abs(differences))[::-1]
    sorted_diffs = [differences[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    plt.barh(y_pos, sorted_diffs, align='center', color=sorted_colors)
    plt.yticks(y_pos, sorted_labels)
    plt.xlabel('Percentage Difference (Successful vs Unsuccessful)')
    plt.title('Relative Difference in Language Pattern Usage')
    
    # Add a line at 0%
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, v in enumerate(sorted_diffs):
        plt.text(v + np.sign(v) * 3, i, f"{v:.1f}%", 
                 va='center', ha='left' if v >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_normalized_sentiment_patterns(successful_sentiment, unsuccessful_sentiment, output_file='output/normalized_sentiment_patterns.png'):
    """Plot normalized sentiment pattern comparison showing relative differences."""
    if not successful_sentiment and not unsuccessful_sentiment:
        print("Warning: No sentiment patterns to plot for normalization")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate percentage differences for all metrics
    metrics = ['polarity', 'subjectivity', 'positive_sentences', 'negative_sentences', 'neutral_sentences']
    differences = []
    labels = []
    colors = []
    
    for metric in metrics:
        s_value = successful_sentiment.get(metric, 0)
        u_value = unsuccessful_sentiment.get(metric, 0)
        
        # Calculate percentage difference relative to unsuccessful
        if u_value != 0:
            diff_percent = ((s_value - u_value) / u_value) * 100
        else:
            diff_percent = 100 if s_value > 0 else 0
            
        differences.append(diff_percent)
        labels.append(metric.replace('_', ' ').title())
        colors.append('#4CAF50' if diff_percent >= 0 else '#F44336')
    
    # Sort by absolute percentage difference
    sorted_indices = np.argsort(np.abs(differences))[::-1]
    sorted_diffs = [differences[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    y_pos = np.arange(len(sorted_labels))
    
    # Plot the differences
    ax.barh(y_pos, sorted_diffs, align='center', color=sorted_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Percentage Difference (Successful vs Unsuccessful)')
    ax.set_title('Relative Difference in Sentiment Patterns')
    
    # Add a line at 0%
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, v in enumerate(sorted_diffs):
        ax.text(v + np.sign(v) * 3, i, f"{v:.1f}%", 
                va='center', ha='left' if v >= 0 else 'right')
    
    # Add a small table showing actual values for comparison
    table_data = []
    table_columns = ['Metric', 'Successful', 'Unsuccessful']
    
    for i, metric in enumerate(metrics):
        s_value = successful_sentiment.get(metric, 0)
        u_value = unsuccessful_sentiment.get(metric, 0)
        
        if metric in ['polarity', 'subjectivity']:
            table_data.append([labels[i], f"{s_value:.3f}", f"{u_value:.3f}"])
        else:
            table_data.append([labels[i], f"{s_value:.1f}%", f"{u_value:.1f}%"])
    
    # Add a table at the bottom
    plt.figtext(0.5, 0.01, 'Raw values for reference:', ha='center')
    the_table = plt.table(cellText=table_data, colLabels=table_columns, 
                          loc='bottom', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.2)
    
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_normalized_response_length_by_question(common_questions, output_file='output/normalized_response_length_by_question.png'):
    """Plot normalized response length by question showing percentage differences."""
    if not common_questions:
        print("Warning: No common questions to plot for normalization")
        return

    # Find questions with both successful and unsuccessful responses
    questions_with_both = {}
    for question, data in common_questions.items():
        if data['successful'] and data['unsuccessful']:
            successful_avg = sum(data['successful']) / len(data['successful'])
            unsuccessful_avg = sum(data['unsuccessful']) / len(data['unsuccessful'])
            
            # Calculate percentage difference relative to unsuccessful
            if unsuccessful_avg != 0:
                diff_percent = ((successful_avg - unsuccessful_avg) / unsuccessful_avg) * 100
            else:
                diff_percent = 100 if successful_avg > 0 else 0
                
            questions_with_both[question] = {
                'successful_avg': successful_avg,
                'unsuccessful_avg': unsuccessful_avg,
                'diff_percent': diff_percent
            }

    # Sort by absolute percentage difference
    sorted_questions = sorted(questions_with_both.items(), key=lambda x: abs(x[1]['diff_percent']), reverse=True)

    # Visualize top differentiating questions
    top_n = min(10, len(sorted_questions))
    questions = [q[:40] + '...' if len(q) > 40 else q for q, _ in sorted_questions[:top_n]]
    diff_percents = [data['diff_percent'] for _, data in sorted_questions[:top_n]]
    colors = ['#4CAF50' if d >= 0 else '#F44336' for d in diff_percents]

    plt.figure(figsize=(14, 10))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(questions))
    ax.barh(y_pos, diff_percents, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(questions)
    ax.set_xlabel('Percentage Difference in Response Length (Successful vs Unsuccessful)')
    ax.set_title('Relative Difference in Response Length by Question')
    
    # Add a line at 0%
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, v in enumerate(diff_percents):
        ax.text(v + np.sign(v) * 5, i, f"{v:.1f}% ({int(sorted_questions[i][1]['successful_avg'])} vs {int(sorted_questions[i][1]['unsuccessful_avg'])} words)", 
                va='center', ha='left' if v >= 0 else 'right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_normalized_response_length_distribution(successful_lengths, unsuccessful_lengths, output_file='output/normalized_response_length_distribution.png'):
    """Plot normalized response length distribution showing relative distributions."""
    if not successful_lengths and not unsuccessful_lengths:
        print("Warning: No response lengths to plot for distribution")
        return
    
    # Extract lengths from data
    s_lengths = [item['length'] for item in successful_lengths]
    u_lengths = [item['length'] for item in unsuccessful_lengths]
    
    # Create DataFrame for easier plotting
    data = []
    for length in s_lengths:
        data.append({'length': length, 'group': 'Successful'})
    for length in u_lengths:
        data.append({'length': length, 'group': 'Unsuccessful'})
    
    df = pd.DataFrame(data)
    
    # Calculate statistics
    s_mean = np.mean(s_lengths)
    u_mean = np.mean(u_lengths)
    s_median = np.median(s_lengths)
    u_median = np.median(u_lengths)
    
    # Create figure with multiple plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 1. Density plot (normalized histogram)
    sns.histplot(data=df, x='length', hue='group', kde=True, stat='density',
                element='step', palette={'Successful': '#4CAF50', 'Unsuccessful': '#F44336'}, ax=ax1)
    ax1.set_title('Normalized Response Length Distribution')
    ax1.set_xlabel('Response Length (words)')
    ax1.set_ylabel('Density')
    
    # Add vertical lines for means
    ax1.axvline(s_mean, color='#4CAF50', linestyle='--', alpha=0.7, label=f'Successful Mean: {s_mean:.1f}')
    ax1.axvline(u_mean, color='#F44336', linestyle='--', alpha=0.7, label=f'Unsuccessful Mean: {u_mean:.1f}')
    ax1.legend()
    
    # 2. Box plot with violin overlay
    sns.violinplot(data=df, x='group', y='length', palette={'Successful': '#4CAF50', 'Unsuccessful': '#F44336'}, ax=ax2)
    sns.boxplot(data=df, x='group', y='length', width=0.3, boxprops={'alpha': 0.3}, ax=ax2)
    ax2.set_title('Response Length Comparison (Box and Violin Plot)')
    ax2.set_xlabel('Application Status')
    ax2.set_ylabel('Response Length (words)')
    
    # Add statistical annotations
    stats_text = (f"Successful: mean={s_mean:.1f}, median={s_median:.1f}, n={len(s_lengths)}\n"
                 f"Unsuccessful: mean={u_mean:.1f}, median={u_median:.1f}, n={len(u_lengths)}")
    ax2.text(0.5, 0.01, stats_text, transform=ax2.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close() 