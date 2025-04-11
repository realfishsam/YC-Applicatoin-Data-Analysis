# YC Application Data Analysis

This project analyzes YC (Y Combinator) application data to understand patterns and differences between successful and unsuccessful applications.

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw HTML data from getintoyc.com
│   │   ├── companies_htmls/
│   │   └── yc_applications.html
│   ├── processed/        # Processed JSON data
│   └── nltk_data/       # NLTK resources
├── output/              # Generated visualizations and analysis results
├── src/                # Source code
│   ├── data_downloader.py  # Script to download raw data
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── text_analysis.py    # Text analysis functions
│   ├── visualization.py    # Visualization functions
│   └── main.py            # Main analysis script
└── requirements.txt    # Python dependencies
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the data:
   ```bash
   python src/data_downloader.py
   ```
   This will create the necessary directories and download company application data.

3. Run the analysis:
   ```bash
   python src/main.py
   ```
   This will generate visualizations and analysis results in the `output` directory.

## Analysis Components

- **Data Collection**: Uses Selenium to scrape YC application data from getintoyc.com
- **Text Analysis**: 
  - Word frequency analysis
  - Language pattern analysis
  - Response length analysis
- **Visualizations**:
  - Word clouds comparing successful vs unsuccessful applications
  - Top words comparison
  - Language pattern comparisons
  - Response length distributions

## Dependencies

- Python 3.8+
- See `requirements.txt` for complete list of Python packages

## Notes

- The data is scraped from getintoyc.com, which contains publicly available YC applications
- NLTK resources are downloaded locally to `data/nltk_data/`
- Generated visualizations are saved in the `output/` directory
- Raw HTML data is stored in `data/raw/` and processed JSON in `data/processed/`