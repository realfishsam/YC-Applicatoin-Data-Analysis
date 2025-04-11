import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
import sys # Import sys for exiting
import os # Import os for directory creation

# Define the local NLTK data directory
NLTK_DATA_DIR = './nltk_data'

# Create the directory if it doesn't exist
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

# Ensure NLTK resources are downloaded before importing them
try:
    # Download necessary NLTK resources to the local directory
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=False)
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=False)
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=False)
except Exception as e:
    print(f"Error downloading NLTK data: {e}", file=sys.stderr)
    print("Please check your internet connection and ensure you don't have firewall/proxy issues.", file=sys.stderr)
    print(f"Attempted to download to: {os.path.abspath(NLTK_DATA_DIR)}", file=sys.stderr)
    sys.exit(1) # Exit if download fails
    
# Add the local directory to NLTK's data path
nltk.data.path.append(NLTK_DATA_DIR)

# Now import the resources (should find them in the local path)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Load the data
with open('data_processed/company_qna_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Separate successful and unsuccessful applications
successful_companies = [company for company in data if company['status'] == 'Successful']
unsuccessful_companies = [company for company in data if company['status'] == 'Unsuccessful']

print(f"Number of successful applications: {len(successful_companies)}")
print(f"Number of unsuccessful applications: {len(unsuccessful_companies)}")

# Extract all QnA pairs
def extract_qna(companies):
    all_qna = []
    for company in companies:
        for qa in company['qna']:
            all_qna.append({
                'company': company['company_id'],
                'question': qa['question'],
                'answer': qa['answer'],
                'status': company['status']
            })
    return all_qna

all_qna = extract_qna(data)
successful_qna = [qa for qa in all_qna if qa['status'] == 'Successful']
unsuccessful_qna = [qa for qa in all_qna if qa['status'] == 'Unsuccessful']

print(f"Total QnA pairs: {len(all_qna)}")
print(f"Successful QnA pairs: {len(successful_qna)}")
print(f"Unsuccessful QnA pairs: {len(unsuccessful_qna)}")

# Convert to DataFrame for easier manipulation
qa_df = pd.DataFrame(all_qna)

# Combine all answers for each group
successful_text = ' '.join([qa['answer'] for qa in successful_qna])
unsuccessful_text = ' '.join([qa['answer'] for qa in unsuccessful_qna])

# Basic statistics
print(f"Total words in successful applications: {len(successful_text.split())}")
print(f"Total words in unsuccessful applications: {len(unsuccessful_text.split())}")
print(f"Average words per answer in successful applications: {len(successful_text.split()) / len(successful_qna):.2f}")
print(f"Average words per answer in unsuccessful applications: {len(unsuccessful_text.split()) / len(unsuccessful_qna):.2f}")

# Function to clean text and get word frequencies
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize (NLTK download must have succeeded)
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def get_word_frequencies(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters (keep spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    words = text.split()
    
    # Use NLTK stopwords - download must succeed
    stop_words = set(stopwords.words('english'))
    # Removed LookupError handling here
        
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2 and not word.isdigit()]
    # Count frequencies
    return Counter(filtered_words)

successful_freq = get_word_frequencies(successful_text)
unsuccessful_freq = get_word_frequencies(unsuccessful_text)

# Normalize by total word count
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
    
    # Only consider words that appear at least a few times
    successful_count = successful_freq.get(word, 0)
    unsuccessful_count = unsuccessful_freq.get(word, 0)
    
    if (successful_count >= 3 or unsuccessful_count >= 3) and abs(diff) > 0.0005:
        diff_words[word] = {
            'successful_freq': s_freq_norm,
            'unsuccessful_freq': u_freq_norm,
            'difference': diff,
            'successful_count': successful_count,
            'unsuccessful_count': unsuccessful_count
        }

# Calculate response lengths
def get_response_lengths(qna_list):
    return [{
        'company': qa['company'],
        'question': qa['question'],
        'length': len(qa['answer'].split())
    } for qa in qna_list]

successful_lengths = get_response_lengths(successful_qna)
unsuccessful_lengths = get_response_lengths(unsuccessful_qna)

# Calculate average response length by company
def get_company_avg_lengths(lengths):
    df = pd.DataFrame(lengths)
    return df.groupby('company')['length'].mean().to_dict()

successful_avg_by_company = get_company_avg_lengths(successful_lengths)
unsuccessful_avg_by_company = get_company_avg_lengths(unsuccessful_lengths)

# Visualize word frequencies
plt.figure(figsize=(12, 10))

# Top words in successful applications
successful_top_words = dict(sorted(successful_freq.items(), key=lambda x: x[1], reverse=True)[:20])
plt.subplot(2, 1, 1)
sns.barplot(x=list(successful_top_words.values()), y=list(successful_top_words.keys()), palette='viridis')
plt.title('Top 20 Words in Successful Applications')
plt.xlabel('Frequency')

# Top words in unsuccessful applications
unsuccessful_top_words = dict(sorted(unsuccessful_freq.items(), key=lambda x: x[1], reverse=True)[:20])
plt.subplot(2, 1, 2)
sns.barplot(x=list(unsuccessful_top_words.values()), y=list(unsuccessful_top_words.keys()), palette='viridis')
plt.title('Top 20 Words in Unsuccessful Applications')
plt.xlabel('Frequency')

plt.tight_layout()
plt.savefig('top_words_comparison.png', dpi=300)

# Visualize word clouds
plt.figure(figsize=(18, 8))

# Word cloud for successful applications
plt.subplot(1, 2, 1)
successful_clean_text = clean_text(successful_text)
successful_wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                 max_words=100, colormap='viridis').generate(successful_clean_text)
plt.imshow(successful_wordcloud, interpolation='bilinear')
plt.title('Words in Successful Applications', fontsize=20)
plt.axis('off')

# Word cloud for unsuccessful applications
plt.subplot(1, 2, 2)
unsuccessful_clean_text = clean_text(unsuccessful_text)
unsuccessful_wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                   max_words=100, colormap='viridis').generate(unsuccessful_clean_text)
plt.imshow(unsuccessful_wordcloud, interpolation='bilinear')
plt.title('Words in Unsuccessful Applications', fontsize=20)
plt.axis('off')

plt.tight_layout()
plt.savefig('wordcloud_comparison.png', dpi=300)

# Analyze language patterns
def analyze_language_patterns(text):
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
    
    # Normalize by word count
    word_count = len(text.split())
    normalized_patterns = {k: v / word_count * 1000 for k, v in patterns.items()}  # Per 1000 words
    
    return normalized_patterns

successful_patterns = analyze_language_patterns(successful_text)
unsuccessful_patterns = analyze_language_patterns(unsuccessful_text)

# Visualize language patterns
plt.figure(figsize=(14, 8))
patterns = list(successful_patterns.keys())
successful_values = [successful_patterns[p] for p in patterns]
unsuccessful_values = [unsuccessful_patterns[p] for p in patterns]

x = np.arange(len(patterns))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, successful_values, width, label='Successful', color='#4CAF50')
rects2 = ax.bar(x + width/2, unsuccessful_values, width, label='Unsuccessful', color='#F44336')

ax.set_ylabel('Frequency per 1,000 words')
ax.set_title('Language Pattern Usage Comparison')
ax.set_xticks(x)
ax.set_xticklabels([p.replace('firstPerson', '1st Person ').replace('S', ' S') for p in patterns])
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('language_patterns.png', dpi=300)

# Analyze response length for common questions
common_questions = {}
for qa in all_qna:
    question = qa['question']
    status = qa['status']
    length = len(qa['answer'].split())
    
    if question not in common_questions:
        common_questions[question] = {'successful': [], 'unsuccessful': []}
    
    if status == 'Successful':
        common_questions[question]['successful'].append(length)
    else:
        common_questions[question]['unsuccessful'].append(length)

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
top_n = 10
plt.figure(figsize=(14, 10))

questions = [q[:40] + '...' if len(q) > 40 else q for q, _ in sorted_questions[:top_n]]
successful_avgs = [data['successful_avg'] for _, data in sorted_questions[:top_n]]
unsuccessful_avgs = [data['unsuccessful_avg'] for _, data in sorted_questions[:top_n]]

x = np.arange(len(questions))
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

fig.tight_layout()
plt.savefig('response_length_by_question.png', dpi=300)

# Create a company length comparison chart
company_lengths = []

for company_id, avg_length in successful_avg_by_company.items():
    company_lengths.append({
        'name': company_id,
        'status': 'Successful',
        'avgLength': avg_length
    })

for company_id, avg_length in unsuccessful_avg_by_company.items():
    company_lengths.append({
        'name': company_id,
        'status': 'Unsuccessful',
        'avgLength': avg_length
    })

company_lengths_sorted = sorted(company_lengths, key=lambda x: x['avgLength'], reverse=True)

plt.figure(figsize=(14, 8))
companies = [item['name'] for item in company_lengths_sorted]
lengths = [item['avgLength'] for item in company_lengths_sorted]
colors = ['#4CAF50' if item['status'] == 'Successful' else '#F44336' for item in company_lengths_sorted]

plt.bar(companies, lengths, color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Words per Answer')
plt.title('Average Response Length by Company')

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', label='Successful'),
    Patch(facecolor='#F44336', label='Unsuccessful')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('company_response_lengths.png', dpi=300)

# Print summary of key findings
print("\n--- KEY FINDINGS ---")
print("\n1. Language Pattern Usage:")
for pattern in successful_patterns:
    print(f"  {pattern}: Successful={successful_patterns[pattern]:.2f}, Unsuccessful={unsuccessful_patterns[pattern]:.2f}, Diff={successful_patterns[pattern]-unsuccessful_patterns[pattern]:.2f}")

print("\n2. Top Differentiating Words (Successful Applications Use More):")
diff_words_sorted = sorted(diff_words.items(), key=lambda x: x[1]['difference'], reverse=True)
for word, data in diff_words_sorted[:10]:
    print(f"  {word}: +{data['difference']*100:.2f}% ({int(data['successful_count'])} vs {int(data['unsuccessful_count'])} occurrences)")

print("\n3. Top Differentiating Words (Unsuccessful Applications Use More):")
diff_words_sorted = sorted(diff_words.items(), key=lambda x: x[1]['difference'])
for word, data in diff_words_sorted[:10]:
    print(f"  {word}: {data['difference']*100:.2f}% ({int(data['successful_count'])} vs {int(data['unsuccessful_count'])} occurrences)")

print("\n4. Questions with Biggest Response Length Differences:")
for question, data in sorted_questions[:5]:
    print(f"  \"{question[:50]}...\"")
    print(f"    Successful avg: {data['successful_avg']:.1f} words")
    print(f"    Unsuccessful avg: {data['unsuccessful_avg']:.1f} words")
    print(f"    Difference: {data['difference']:.1f} words\n")

def calculate_question_success_ratio(all_qna):
    questions = {}
    for qa in all_qna:
        question = qa['question']
        status = qa['status']
        length = len(qa['answer'].split())
        
        if question not in questions:
            questions[question] = {'successful': [], 'unsuccessful': []}
        
        if status == 'Successful':
            questions[question]['successful'].append(length)
        else:
            questions[question]['unsuccessful'].append(length)
    
    # Calculate average successful and unsuccessful lengths for each question
    for question, data in questions.items():
        successful_avg = sum(data['successful']) / len(data['successful']) if data['successful'] else 0
        unsuccessful_avg = sum(data['unsuccessful']) / len(data['unsuccessful']) if data['unsuccessful'] else 0
        questions[question] = {
            'avg_successful': successful_avg,
            'avg_unsuccessful': unsuccessful_avg,
            'ratio': successful_avg / unsuccessful_avg if unsuccessful_avg else float('inf')
        }
    
    # Sort by ratio
    questions = dict(sorted(questions.items(), key=lambda x: x[1]['ratio'], reverse=True))
    return questions

def analyze_by_batch(data):
    # Group companies by batch
    batch_groups = {}
    for company in data:
        batch = company.get('batch', 'Unknown') # Use .get for safety
        if batch not in batch_groups:
            batch_groups[batch] = []
        batch_groups[batch].append(company)
    
    # Analyze patterns for each batch
    batch_results = {}
    for batch, companies in batch_groups.items():
        successful = [c for c in companies if c['status'] == 'Successful']
        unsuccessful = [c for c in companies if c['status'] == 'Unsuccessful']
        
        batch_results[batch] = {
            'total': len(companies),
            'successful_count': len(successful),
            'unsuccessful_count': len(unsuccessful),
            'success_rate': len(successful) / len(companies) * 100 if companies else 0, # As percentage
            # TODO: Add more metrics here if needed (e.g., avg answer length per batch)
        }
    
    # Sort results by batch name for consistent output
    return dict(sorted(batch_results.items()))


# --- Run Analyses ---

# Original Language Patterns
# ... existing code ...
# Question Success Ratios (based on length)
question_ratios = calculate_question_success_ratio(all_qna)

# Batch Analysis
batch_analysis_results = analyze_by_batch(data) # Use original data list

# Combine patterns for visualization and reporting (Excluding raw team counts)
# ... existing code ...

# Print Question Success Ratio Analysis
print("\n6. Question Response Length Ratio (Successful Avg / Unsuccessful Avg):")
# ... existing code ...
    print(f"  {ratio_str:<6} | {data['avg_successful']:<5.0f} | {data['avg_unsuccessful']:<5.0f} | \"{question[:60]}...\"")

# Print Batch Analysis Results
print("\n7. Analysis by Batch:")
print("  Batch     | Total | Successful | Unsuccessful | Success Rate")
print("  ----------|-------|------------|--------------|--------------")
for batch, results in batch_analysis_results.items():
    print(f"  {batch:<9} | {results['total']:>5} | {results['successful_count']:>10} | {results['unsuccessful_count']:>12} | {results['success_rate']:>12.1f}%")