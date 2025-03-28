import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from config import COMPETITOR_DOMAINS

class CompetitorScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_page_content(self, url: str) -> Dict:
        """Extract clean text content from a competitor URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            return {
                'url': url,
                'text': soup.get_text(separator='\n', strip=True),
                'html': str(soup)
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def batch_scrape(self, urls: List[str]) -> List[Dict]:
        """Process multiple competitor URLs"""
        return [self.extract_page_content(url) for url in urls if url] 