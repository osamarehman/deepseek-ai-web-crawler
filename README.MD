# SEO Optimization Scraper

## Objective
Analyze website content for SEO optimization by extracting:
- Heading hierarchy (H1-H6)
- Content structure
- Internal linking patterns
- Keyword usage
- Content quality metrics
- Generate actionable SEO recommendations

## Features
1. **Core Analysis**
- Heading structure validation
- Content length analysis
- Internal link mapping
- Keyword density tracking
- Missing alt text detection
- Page speed indicators

2. **AI-Powered Recommendations**
- Content restructuring suggestions
- Semantic keyword ideas
- Heading optimization
- Meta tag improvements
- Content gap analysis

3. **Reporting**
- CSV exports with detailed findings
- Executive summary reports
- Priority-based improvement list
- Competitor comparison metrics

## New Features

### Content Analysis
- Full HTML tag hierarchy mapping
- Text content segmentation by section
- CTA effectiveness scoring
- Content gap analysis against competitors

### Keyword Integration
```tsv
Format:
Keyword [TAB] Type (Primary/Secondary) [TAB] Monthly Volume [TAB] 
Difficulty [TAB] Importance [TAB] Target Pages (comma-separated)
```

### Outputs
1. `seo_audit.csv` - Detailed technical/content analysis
2. `conversion_analysis.csv` - CTA performance and suggestions
3. `components.html` - WordPress-ready HTML snippets

## Project Structure
```
.
├── main.py # Main entry point for the crawler
├── config.py # Contains configuration constants (Base URL, CSS selectors, etc.)
├── models
│ └── venue.py # Defines the Venue data model using Pydantic
├── utils
│ ├── init.py # (Empty) Package marker for utils
│ ├── data_utils.py # Utility functions for processing and saving data
│ └── scraper_utils.py # Utility functions for configuring and running the crawler
├── requirements.txt # Python package dependencies
├── .gitignore # Git ignore file (e.g., excludes .env and CSV files)
└── README.MD # This file
```

## Installation

1. **Create and Activate a Conda Environment**

   ```bash
   conda create -n deep-seek-crawler python=3.12 -y
   conda activate deep-seek-crawler
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Your Environment Variables**

   Create a `.env` file in the root directory with content similar to:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

   *(Note: The `.env` file is in your .gitignore, so it won't be pushed to version control.)*

## Usage

To start the crawler, run:

```bash
python main.py
```

The script will crawl the specified website, extract data page by page, and save the complete venues to a `complete_venues.csv` file in the project directory. Additionally, usage statistics for the LLM strategy will be displayed after crawling.

## Configuration

The `config.py` file contains key constants used throughout the project:

- **BASE_URL**: The URL of the website from which to extract venue data.
- **CSS_SELECTOR**: CSS selector string used to target venue content.
- **REQUIRED_KEYS**: List of required fields to consider a venue complete.

You can modify these values as needed.

## Additional Notes

- **Logging:** The project currently uses print statements for status messages. For production or further development, consider integrating Python's built-in `logging` module.
- **Improvements:** The code is structured in multiple modules to maintain separation of concerns, making it easier for beginners to follow and extend the functionality.
- **Dependencies:** Ensure that the package versions specified in `requirements.txt` are installed to avoid compatibility issues.

## License

Include license information if applicable.

## WordPress Integration

### Template Types
1. **Landing Pages**
   - Service-focused (e.g., "/fue-hair-transplant")
   - Location-focused (e.g., "/new-york-hair-transplant")
2. **Service Pages**
   - Detailed procedure explanations
   - Before/after galleries
3. **Blog Components**
   - FAQ sections
   - Comparison tables
   - Expert opinion blocks

### Usage
```python
from wordpress_templates import PageTemplateGenerator

generator = PageTemplateGenerator(analysis_data)
landing_page = generator.generate_landing_page()
with open("new_york_fue.html", "w") as f:
    f.write(landing_page)
```
