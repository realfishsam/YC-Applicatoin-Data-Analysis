import os
import json
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

class YCApplicationDownloader:
    def __init__(self):
        self.base_url = "https://getintoyc.com/"
        self.wait_text = "Code For Cash"
        self.wait_timeout = 30
        self.output_dir = "data/raw"
        
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(os.path.join(self.output_dir, "companies_htmls"), exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
    
    def download_main_page(self):
        """Download the main page containing company links."""
        print("Downloading main page...")
        driver = None
        try:
            driver = webdriver.Chrome()
            driver.get(self.base_url)
            
            # Wait for dynamic content
            wait_xpath = f"//*[contains(text(), '{self.wait_text}')]"
            wait_condition = EC.visibility_of_element_located((By.XPATH, wait_xpath))
            WebDriverWait(driver, self.wait_timeout).until(wait_condition)
            
            # Save HTML
            html = driver.page_source
            main_page_path = os.path.join(self.output_dir, "yc_applications.html")
            with open(main_page_path, "w", encoding="utf-8") as f:
                f.write(html)
                
            return html
            
        finally:
            if driver:
                driver.quit()
    
    def extract_company_links(self, html):
        """Extract company links from the main page HTML."""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for link in soup.find_all("a", class_="ct-link brand-color company"):
            links.append(link["href"])
        return links
    
    def download_company_page(self, url):
        """Download individual company page."""
        company_id = url.split('/')[-2]
        output_file = os.path.join(self.output_dir, "companies_htmls", f"{company_id}.html")
        
        if os.path.exists(output_file):
            print(f"Skipping {company_id} - already downloaded")
            return
        
        print(f"Processing: {company_id} ({url})")
        driver = None
        try:
            driver = webdriver.Chrome()
            driver.get(url)
            
            # Wait for batch information to appear
            wait_xpath = "//*[contains(text(), 'Batch')]"
            wait_condition = EC.visibility_of_element_located((By.XPATH, wait_xpath))
            WebDriverWait(driver, self.wait_timeout).until(wait_condition)
            
            # Save HTML
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
                
        except Exception as e:
            print(f"Error processing {company_id}: {e}")
            
        finally:
            if driver:
                driver.quit()
    
    def process_company_data(self, html_file):
        """Extract relevant information from company HTML file."""
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        company_id = os.path.splitext(os.path.basename(html_file))[0]
        
        # Extract batch and status
        batch_text = soup.find(string=lambda t: t and 'Batch:' in t)
        batch = batch_text.strip() if batch_text else None
        
        status_text = soup.find(string=lambda t: t and ('Successful' in t or 'Unsuccessful' in t))
        status = status_text.strip() if status_text else None
        
        # Extract Q&A pairs from the card structure
        qna_pairs = []
        cards = soup.find_all('div', class_='card')
        
        for card in cards:
            question_div = card.find('div', class_='card-top')
            answer_div = card.find('div', class_='card-bottom')
            
            if question_div and answer_div:
                question_link = question_div.find('a')
                answer_text = answer_div.find('span')
                
                if question_link and answer_text:
                    qna_pairs.append({
                        'question': question_link.get_text(strip=True),
                        'answer': answer_text.get_text(strip=True)
                    })
        
        print(f"Found {len(qna_pairs)} Q&A pairs for {company_id}")
        
        return {
            'company_id': company_id,
            'batch': batch,
            'status': status,
            'qna': qna_pairs
        }
    
    def run(self):
        """Run the complete download and processing pipeline."""
        self.setup_directories()
        
        # Download main page and extract company links
        html = self.download_main_page()
        company_links = self.extract_company_links(html)
        print(f"Found {len(company_links)} company links")
        
        # Download individual company pages
        for url in company_links:
            self.download_company_page(url)
            time.sleep(1)  # Be nice to the server
        
        # Process all company data
        all_companies = []
        companies_dir = os.path.join(self.output_dir, "companies_htmls")
        for html_file in os.listdir(companies_dir):
            if html_file.endswith('.html'):
                company_data = self.process_company_data(os.path.join(companies_dir, html_file))
                if company_data['status']:  # Only include if status was found
                    all_companies.append(company_data)
        
        # Save processed data
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        output_file = os.path.join(processed_dir, "company_qna_data.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_companies, f, indent=2)
        
        print(f"Processed data saved to {output_file}")
        print(f"Total companies processed: {len(all_companies)}")

    def process_only(self):
        """Only process existing HTML files without downloading new data."""
        self.setup_directories()
        
        # Process all company data
        all_companies = []
        companies_dir = os.path.join(self.output_dir, "companies_htmls")
        
        if not os.path.exists(companies_dir):
            print(f"Error: Directory {companies_dir} does not exist!")
            return
            
        print("Processing existing HTML files...")
        for html_file in os.listdir(companies_dir):
            if html_file.endswith('.html'):
                try:
                    company_data = self.process_company_data(os.path.join(companies_dir, html_file))
                    if company_data['status']:  # Only include if status was found
                        all_companies.append(company_data)
                        print(f"Processed: {company_data['company_id']}")
                except Exception as e:
                    print(f"Error processing {html_file}: {e}")
        
        # Save processed data
        output_file = "data/processed/company_qna_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_companies, f, indent=2)
        
        print(f"Processed data saved to {output_file}")
        print(f"Total companies processed: {len(all_companies)}")

if __name__ == "__main__":
    downloader = YCApplicationDownloader()
    downloader.process_only()  # Only run the processing part 