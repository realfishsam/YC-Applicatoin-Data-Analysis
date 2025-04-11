import json
import pandas as pd

def load_data(file_path='data/processed/company_qna_data.json'):
    """Load the QnA data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def separate_by_status(data):
    """Separate successful and unsuccessful applications."""
    successful = [company for company in data if company['status'] == 'Successful']
    unsuccessful = [company for company in data if company['status'] == 'Unsuccessful']
    return successful, unsuccessful

def extract_qna(companies):
    """Extract all QnA pairs from companies data."""
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

def get_response_lengths(qna_list):
    """Calculate response lengths for QnA pairs."""
    return [{
        'company': qa['company'],
        'question': qa['question'],
        'length': len(qa['answer'].split())
    } for qa in qna_list]

def get_company_avg_lengths(lengths):
    """Calculate average response length by company."""
    df = pd.DataFrame(lengths)
    return df.groupby('company')['length'].mean().to_dict()

def prepare_data(data_file='data/processed/company_qna_data.json'):
    """Load and prepare data for analysis."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all QnA pairs
    all_qna = []
    for company in data:
        for qa in company['qna']:
            all_qna.append({
                'company': company['company_id'],
                'question': qa['question'],
                'answer': qa['answer'],
                'status': company['status']
            })
    
    # Separate successful and unsuccessful QnA pairs
    successful_qna = [qa for qa in all_qna if qa['status'] == 'Successful']
    unsuccessful_qna = [qa for qa in all_qna if qa['status'] == 'Unsuccessful']
    
    return {
        'successful_qna': successful_qna,
        'unsuccessful_qna': unsuccessful_qna,
        'all_qna': all_qna
    }

def get_common_questions(all_qna):
    """Get response lengths grouped by question for both successful and unsuccessful applications."""
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
    
    return common_questions 