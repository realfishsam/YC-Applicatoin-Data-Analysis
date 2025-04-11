import os
import pandas as pd

def ensure_export_dir(directory='output/csv'):
    """Ensure the export directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def export_word_frequencies(successful_freq, unsuccessful_freq, output_file='output/csv/word_frequencies.csv'):
    """Export word frequencies to CSV."""
    ensure_export_dir()
    
    # Create a combined DataFrame
    data = []
    
    # Get all unique words
    all_words = set(list(successful_freq.keys()) + list(unsuccessful_freq.keys()))
    
    # Calculate totals for normalization
    successful_total = sum(successful_freq.values())
    unsuccessful_total = sum(unsuccessful_freq.values())
    
    # Add each word to the data
    for word in all_words:
        s_count = successful_freq.get(word, 0)
        u_count = unsuccessful_freq.get(word, 0)
        
        # Calculate normalized frequencies (per 1000 words)
        s_norm = (s_count / successful_total * 1000) if successful_total > 0 else 0
        u_norm = (u_count / unsuccessful_total * 1000) if unsuccessful_total > 0 else 0
        
        # Calculate relative difference
        if u_norm > 0:
            rel_diff = ((s_norm - u_norm) / u_norm) * 100
        elif s_norm > 0:
            rel_diff = 100
        else:
            rel_diff = 0
            
        data.append({
            'word': word,
            'successful_count': s_count,
            'unsuccessful_count': u_count,
            'successful_normalized': s_norm,
            'unsuccessful_normalized': u_norm,
            'normalized_difference': s_norm - u_norm,
            'percentage_difference': rel_diff
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.sort_values('normalized_difference', ascending=False, inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Word frequencies exported to {output_file}")

def export_language_patterns(successful_patterns, unsuccessful_patterns, output_file='output/csv/language_patterns.csv'):
    """Export language patterns to CSV."""
    ensure_export_dir()
    
    # Create a combined DataFrame
    data = []
    
    for pattern in successful_patterns.keys():
        s_value = successful_patterns.get(pattern, 0)
        u_value = unsuccessful_patterns.get(pattern, 0)
        
        # Calculate percentage difference
        if u_value > 0:
            diff_percent = ((s_value - u_value) / u_value) * 100
        elif s_value > 0:
            diff_percent = 100
        else:
            diff_percent = 0
            
        data.append({
            'pattern': pattern,
            'successful_value': s_value,
            'unsuccessful_value': u_value,
            'absolute_difference': s_value - u_value,
            'percentage_difference': diff_percent
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.sort_values('absolute_difference', ascending=False, inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Language patterns exported to {output_file}")

def export_sentiment_patterns(successful_sentiment, unsuccessful_sentiment, output_file='output/csv/sentiment_patterns.csv'):
    """Export sentiment patterns to CSV."""
    ensure_export_dir()
    
    # Create a combined DataFrame
    data = []
    
    for metric in successful_sentiment.keys():
        s_value = successful_sentiment.get(metric, 0)
        u_value = unsuccessful_sentiment.get(metric, 0)
        
        # Calculate percentage difference
        if u_value > 0:
            diff_percent = ((s_value - u_value) / u_value) * 100
        elif s_value > 0:
            diff_percent = 100
        else:
            diff_percent = 0
            
        data.append({
            'metric': metric,
            'successful_value': s_value,
            'unsuccessful_value': u_value,
            'absolute_difference': s_value - u_value,
            'percentage_difference': diff_percent
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Sentiment patterns exported to {output_file}")

def export_response_lengths(successful_lengths, unsuccessful_lengths, output_file='output/csv/response_lengths.csv'):
    """Export response lengths to CSV."""
    ensure_export_dir()
    
    # Extract length values
    s_lengths = [item['length'] for item in successful_lengths]
    u_lengths = [item['length'] for item in unsuccessful_lengths]
    
    s_data = [{'status': 'Successful', 'length': length, 'company': item['company'], 'question': item['question']} 
              for length, item in zip(s_lengths, successful_lengths)]
    
    u_data = [{'status': 'Unsuccessful', 'length': length, 'company': item['company'], 'question': item['question']} 
              for length, item in zip(u_lengths, unsuccessful_lengths)]
    
    # Combine the data
    all_data = s_data + u_data
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Response lengths exported to {output_file}")

def export_response_by_question(common_questions, output_file='output/csv/response_by_question.csv'):
    """Export response lengths by question to CSV."""
    ensure_export_dir()
    
    # Create a combined DataFrame
    data = []
    
    for question, lengths in common_questions.items():
        if lengths['successful'] and lengths['unsuccessful']:
            s_avg = sum(lengths['successful']) / len(lengths['successful'])
            u_avg = sum(lengths['unsuccessful']) / len(lengths['unsuccessful'])
            
            # Calculate percentage difference
            if u_avg > 0:
                diff_percent = ((s_avg - u_avg) / u_avg) * 100
            elif s_avg > 0:
                diff_percent = 100
            else:
                diff_percent = 0
                
            data.append({
                'question': question,
                'successful_avg_length': s_avg,
                'unsuccessful_avg_length': u_avg,
                'successful_count': len(lengths['successful']),
                'unsuccessful_count': len(lengths['unsuccessful']),
                'absolute_difference': s_avg - u_avg,
                'percentage_difference': diff_percent
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.sort_values('absolute_difference', ascending=False, inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Response by question exported to {output_file}")

def export_company_stats(successful_qna, unsuccessful_qna, output_file='output/csv/company_stats.csv'):
    """Export company-level statistics to CSV."""
    ensure_export_dir()
    
    # Group by company
    companies = {}
    
    # Process successful companies
    for qa in successful_qna:
        company = qa['company']
        if company not in companies:
            companies[company] = {
                'company': company,
                'status': 'Successful',
                'total_questions': 0,
                'total_words': 0,
                'response_lengths': []
            }
        
        length = len(qa['answer'].split())
        companies[company]['total_questions'] += 1
        companies[company]['total_words'] += length
        companies[company]['response_lengths'].append(length)
    
    # Process unsuccessful companies
    for qa in unsuccessful_qna:
        company = qa['company']
        if company not in companies:
            companies[company] = {
                'company': company,
                'status': 'Unsuccessful',
                'total_questions': 0,
                'total_words': 0,
                'response_lengths': []
            }
        
        length = len(qa['answer'].split())
        companies[company]['total_questions'] += 1
        companies[company]['total_words'] += length
        companies[company]['response_lengths'].append(length)
    
    # Convert to list of records with calculated metrics
    data = []
    for company, stats in companies.items():
        avg_length = stats['total_words'] / stats['total_questions'] if stats['total_questions'] > 0 else 0
        data.append({
            'company': company,
            'status': stats['status'],
            'total_questions': stats['total_questions'],
            'total_words': stats['total_words'],
            'avg_response_length': avg_length,
            'min_response_length': min(stats['response_lengths']) if stats['response_lengths'] else 0,
            'max_response_length': max(stats['response_lengths']) if stats['response_lengths'] else 0
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.sort_values(['status', 'avg_response_length'], ascending=[True, False], inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Company statistics exported to {output_file}")

def export_all_data(data, successful_freq, unsuccessful_freq, successful_patterns, unsuccessful_patterns, 
                    successful_sentiment, unsuccessful_sentiment, successful_lengths, unsuccessful_lengths, 
                    common_questions):
    """Export all analysis data to CSV files."""
    print("Exporting data to CSV files...")
    
    # Export word frequencies
    export_word_frequencies(successful_freq, unsuccessful_freq)
    
    # Export language patterns
    export_language_patterns(successful_patterns, unsuccessful_patterns)
    
    # Export sentiment patterns
    export_sentiment_patterns(successful_sentiment, unsuccessful_sentiment)
    
    # Export response lengths
    export_response_lengths(successful_lengths, unsuccessful_lengths)
    
    # Export response by question
    export_response_by_question(common_questions)
    
    # Export company statistics
    export_company_stats(data['successful_qna'], data['unsuccessful_qna'])
    
    print("All data exported to CSV files in 'output/csv/' directory") 