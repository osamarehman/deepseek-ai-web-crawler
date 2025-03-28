import json
import os
from typing import List, Set, Tuple, Dict, Optional
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from googlesearch import search

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

from models.venue import Venue
from utils.data_utils import is_complete_venue, is_duplicate_venue
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class SEOAnalysisData(BaseModel):
    # Structural Elements
    all_headings: List[dict] = Field(..., description="All headings H1-H6 with their hierarchy")
    paragraphs: List[str] = Field(..., description="All paragraph texts from main content")
    ctas: List[str] = Field(..., description="All call-to-action button texts and links")
    content_flow: List[dict] = Field(..., description="Section structure with heading/text relationships")
    
    # SEO Metrics
    keyword_usage: dict = Field(..., description="Primary and secondary keyword usage counts")
    content_length: int = Field(..., description="Total word count of meaningful content")
    internal_links: List[dict] = Field(..., description="Internal links with anchor text and destination")
    
    # Competitive Analysis
    competitor_comparison: List[dict] = Field(..., description="Comparison with top 3 competitors")
    
    # Recommendations
    seo_recommendations: List[str] = Field(..., description="Technical and content SEO improvements")
    conversion_recommendations: List[str] = Field(..., description="CTA and layout optimization suggestions")

    # Add to existing model
    page_sections: List[dict] = Field(..., description="Identified page sections with type and content")
    wordpress_components: List[dict] = Field(..., description="Reusable WordPress components")
    strategy_insights: List[str] = Field(..., description="AI-generated content strategy recommendations")


def get_browser_config() -> BrowserConfig:
    """
    Returns the browser configuration for the crawler.

    Returns:
        BrowserConfig: The configuration settings for the browser.
    """
    # https://docs.crawl4ai.com/core/browser-crawler-config/
    return BrowserConfig(
        browser_type="chromium",  # Type of browser to simulate
        headless=False,  # Whether to run in headless mode (no GUI)
        verbose=True,  # Enable verbose logging
    )


def get_llm_strategy(industry: str = "hair-transplant", 
                    focus_keyword: str = "FUE New York") -> LLMExtractionStrategy:
    """
    Returns the configuration for the language model extraction strategy.

    Returns:
        LLMExtractionStrategy: The settings for how to extract data using LLM.
    """
    # https://docs.crawl4ai.com/api/strategies/#llmextractionstrategy
    return LLMExtractionStrategy(
        provider="groq/deepseek-r1-distill-llama-70b",
        api_token=os.getenv("GROQ_API_KEY"),
        schema=SEOAnalysisData.model_json_schema(),
        extraction_type="schema",
        instruction=(
            f"Analyze webpage content for {industry} industry with focus on '{focus_keyword}'. Steps:\n"
            "1. Deconstruct page into sections (hero, features, pricing, etc.)\n"
            "2. Identify content strategy and missing elements\n"
            "3. Compare with top 3 competitors in New York area\n"
            "4. Generate:\n"
            "   - 10 SEO improvements\n"
            "   - 5 conversion optimizations\n"
            "   - 3 content expansion ideas\n"
            "5. Create WordPress-ready component templates\n"
            "Format using markdown tables and bullet points"
        ),
        input_format="html",
        verbose=True,
    )


async def check_no_results(
    crawler: AsyncWebCrawler,
    url: str,
    session_id: str,
) -> bool:
    """
    Checks if the "No Results Found" message is present on the page.

    Args:
        crawler (AsyncWebCrawler): The web crawler instance.
        url (str): The URL to check.
        session_id (str): The session identifier.

    Returns:
        bool: True if "No Results Found" message is found, False otherwise.
    """
    # Fetch the page without any CSS selector or extraction strategy
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            session_id=session_id,
        ),
    )

    if result.success:
        if "No Results Found" in result.cleaned_html:
            return True
    else:
        print(
            f"Error fetching page for 'No Results Found' check: {result.error_message}"
        )

    return False


async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
    url: str,  # Changed from page_number to direct URL
    css_selector: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
) -> dict:
    """Process single URL for SEO analysis"""
    print(f"Analyzing {url}...")
    
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=llm_strategy,
            css_selector=css_selector,
            session_id=session_id,
        ),
    )

    if not result.success:
        print(f"Error analyzing {url}: {result.error_message}")
        return {}

    try:
        analysis_data = json.loads(result.extracted_content)
        analysis_data['url'] = url  # Add source URL
        
        # Post-process recommendations
        if 'recommendations' in analysis_data:
            analysis_data['recommendations'] = [
                rec.replace("**", "").strip() 
                for rec in analysis_data['recommendations']
            ]
            
        # Add content flow analysis
        if 'content_flow' in analysis_data:
            analysis_data['content_flow'] = [
                {**section, 'word_count': len(section.get('content', '').split())}
                for section in analysis_data['content_flow']
            ]
        
        # Add competitor comparison metrics
        analysis_data['competitor_comparison'] = await analyze_competitors(url)
        
        return analysis_data
        
    except json.JSONDecodeError:
        print(f"Failed to parse analysis data for {url}")
        return {}


def export_seo_report(data: List[dict], filename: str = "seo_analysis.csv"):
    """Exports SEO analysis data to CSV with strategy columns"""
    keys = [
        'url', 'primary_keyword', 'content_length', 
        'heading_hierarchy', 'cta_effectiveness',
        'competitor_comparison', 'content_gaps',
        'recommended_actions', 'wordpress_components'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        
        for item in data:
            writer.writerow({
                'url': item.get('url', ''),
                'primary_keyword': item.get('keyword_usage', {}).get('primary', ''),
                'content_length': item.get('content_length', 0),
                'heading_hierarchy': json.dumps(item.get('all_headings', [])),
                'cta_effectiveness': json.dumps(item.get('ctas', [])),
                'competitor_comparison': json.dumps(item.get('competitor_comparison', [])),
                'content_gaps': json.dumps(item.get('content_gaps', [])),
                'recommended_actions': '|'.join(item.get('seo_recommendations', [])),
                'wordpress_components': self._generate_component_links(item)
            })

def _generate_component_links(self, item: dict) -> str:
    """Generates markup for WordPress component links"""
    return ','.join(
        f'<a href="#{comp["type"]}">{comp["type"].title()} Section</a>'
        for comp in item.get('wordpress_components', [])
    )


def export_html_components(data: dict, filename: str = "components.html"):
    """Exports SEO recommendations as WordPress-ready HTML"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("<!-- SEO Optimized Components -->\n")
        
        # Headings section
        f.write("<div class='seo-section'>\n")
        f.write("<h2>Recommended Headings</h2>\n")
        for heading in data.get('all_headings', []):
            f.write(f"<{heading['tag']} class='seo-heading'>{heading['text']}</{heading['tag']}>\n")
        
        # CTAs section
        f.write("<div class='cta-container'>\n")
        for cta in data.get('ctas', []):
            f.write(f"""<a href="{cta['link']}" class="optimized-cta">
                      {cta['text']}</a>\n""")
        f.write("</div></div>")


async def analyze_competitors(target_url: str, competitor_urls: List[str]) -> List[dict]:
    """Analyzes competitor websites for content strategy"""
    crawler = AsyncWebCrawler()
    comparisons = []
    
    for competitor in competitor_urls:
        print(f"Analyzing competitor: {competitor}")
        data = await fetch_and_process_page(
            crawler=crawler,
            url=competitor,
            css_selector="body",
            llm_strategy=get_llm_strategy(),
            session_id="competitor-analysis"
        )
        
        if data:
            comparisons.append({
                'url': competitor,
                'primary_keywords': data.get('keyword_usage', {}).get('primary', []),
                'content_structure': data.get('content_flow', []),
                'top_ctas': data.get('ctas', [])[:3]
            })
    
    return [{
        'target_url': target_url,
        'competitors': comparisons,
        'content_gaps': find_content_gaps(comparisons),
        'opportunities': find_opportunities(comparisons)
    }]

def load_fut_keywords(file_path: str) -> pd.DataFrame:
    """Load and preprocess FUT keywords"""
    df = pd.read_csv(file_path)
    
    # Clean and filter keywords
    df = df[df['Difficulty'] <= 8]  # Focus on achievable keywords
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    
    return df.sort_values(by=['Volume', 'Difficulty'], ascending=[False, True])

def get_secondary_keywords(df: pd.DataFrame, primary: str, count: int=5) -> List[str]:
    """Get related secondary keywords for a primary keyword"""
    filtered = df[df['Parent Keyword'].str.contains(primary, case=False, na=False)]
    return filtered['Keyword'].head(count).tolist()

def fetch_sitemap(url: str) -> List[str]:
    """Fetch and parse XML sitemap with support for indexes"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        
        # Check if it's a sitemap index
        if soup.find('sitemapindex'):
            nested_sitemaps = [loc.text for loc in soup.find_all('loc')]
            all_urls = []
            for nested_url in nested_sitemaps:
                try:
                    all_urls.extend(fetch_sitemap(nested_url))
                except Exception as e:
                    print(f"Error processing nested sitemap {nested_url}: {e}")
            return all_urls
            
        # Process regular sitemap
        return [loc.text for loc in soup.find_all('loc')]
    
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def extract_content(url: str) -> Optional[Dict]:
    """Extract relevant content from URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for elem in soup(['script', 'style', 'nav', 'footer', 'header']):
            elem.decompose()
        
        # Extract structured content
        content = {
            'title': soup.title.string if soup.title else '',
            'headings': [
                {'level': int(h.name[1]), 'text': h.text.strip()}
                for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            ],
            'paragraphs': [p.text.strip() for p in soup.find_all('p')],
            'lists': [
                [li.text.strip() for li in ul.find_all('li')]
                for ul in soup.find_all(['ul', 'ol'])
            ]
        }
        
        return content
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

def analyze_competitor_content(competitor_data: List[Dict]) -> Dict:
    """Analyze competitor content for insights"""
    analysis = {
        'common_sections': [],
        'unique_selling_points': [],
        'content_structure': [],
        'keyword_usage': {},
        'content_gaps': [],
        'recommendations': []
    }
    
    # Analyze common sections and structure
    for competitor in competitor_data:
        for page in competitor['pages']:
            # Analyze headings structure
            headings = page['content'].get('headings', [])
            for heading in headings:
                section = {
                    'title': heading['text'],
                    'level': heading['level']
                }
                if section not in analysis['content_structure']:
                    analysis['content_structure'].append(section)
    
    return analysis

def generate_optimized_content(
    location: str,
    service: str,
    competitor_analysis: Dict,
    requirements: Dict
) -> Dict:
    """Generate optimized page content"""
    content = {
        'title': f"{service} in {location}",
        'meta_description': f"Expert {service} services in {location}. Learn about our advanced techniques, recovery process, and schedule your consultation today.",
        'sections': []
    }
    
    # Generate sections based on competitor analysis
    for section in competitor_analysis['content_structure']:
        if section['level'] == 2:  # Main sections
            content['sections'].append({
                'title': section['title'],
                'content': f"Content for {section['title']} section...",
                'subsections': []
            })
    
    return content

def create_wp_page(content: Dict, template_config: Dict) -> str:
    """Create WordPress page HTML"""
    # Implement WordPress page generation
    html = f"""
    <!-- wp:group {{"className":"location-service-page"}} -->
    <div class="wp-block-group location-service-page">
        <!-- Add page content here -->
    </div>
    <!-- /wp:group -->
    """
    
    return html

def generate_ai_copy(llm, template: str, **kwargs) -> str:
    """Generate content using LLM"""
    prompt = PromptTemplate(
        template=template,
        input_variables=list(kwargs.keys())
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(kwargs)

async def analyze_and_generate_copy(llm, content: str, analysis: dict, config: dict) -> dict:
    """Generate improved copy using AI analysis"""
    prompt = f"""
    Improve this content based on SEO analysis:
    Current content: {content}
    Analysis: {json.dumps(analysis)}
    Requirements: {json.dumps(config)}
    """
    
    improved = generate_ai_copy(llm, prompt)
    
    return {
        'original': content,
        'improved': improved,
        'changes': diff_content(content, improved)
    }

def diff_content(original: str, improved: str) -> str:
    """Generate diff between original and improved content"""
    # Implementation for text comparison
    return "Diff output would be here"

def get_content_keywords(df: pd.DataFrame, primary: str) -> Dict:
    """Get optimized keyword set for content generation"""
    return {
        'primary': primary,
        'secondary': get_secondary_keywords(df, primary),
        'related': df[
            (df['Difficulty'] <= 5) & 
            (df['Volume'] >= 1000)
        ]['Keyword'].tolist()[:10]
    }

def structure_content(raw_content: str) -> List[Dict]:
    """Convert AI-generated text into structured sections"""
    sections = []
    current_heading = None
    
    for line in raw_content.split('\n'):
        if line.startswith('# '):
            sections.append({
                'heading': 'H1',
                'title': line[2:].strip(),
                'content': []
            })
            current_heading = sections[-1]
        elif line.startswith('## '):
            sections.append({
                'heading': 'H2',
                'title': line[3:].strip(),
                'content': []
            })
            current_heading = sections[-1]
        elif current_heading:
            current_heading['content'].append(line.strip())
    
    # Convert content lists to paragraphs
    for section in sections:
        section['content'] = '\n\n'.join(section['content'])
    
    return sections

def web_search(query: str, num_results: int = 5) -> List[Dict]:
    """Perform web search and return structured results"""
    results = []
    try:
        for url in search(query, num_results=num_results, advanced=True):
            results.append({
                'title': url.title,
                'url': url.url,
                'description': url.description
            })
    except Exception as e:
        print(f"Web search error: {e}")
    return results

async def ai_web_analysis(llm, query: str) -> Dict:
    """Get AI-powered web analysis for a query"""
    prompt = f"""
    Analyze the top web results for: {query}
    Identify:
    1. 3 common content patterns
    2. Top 5 keywords used
    3. Content structure best practices
    4. Missing opportunities
    Format as JSON
    """
    return json.loads(generate_ai_copy(llm, prompt))

async def get_competitor_content(domain: str, path: str = None) -> Dict:
    """Fetch and analyze competitor content with enhanced sitemap processing"""
    print(f"\nAnalyzing competitor: {domain}")
    
    # Use specific post sitemap
    sitemap_url = "https://nashvillehairdoctor.com/post-sitemap.xml"
    try:
        urls = fetch_sitemap(sitemap_url)
        print(f"Found {len(urls)} URLs in sitemap")
        
        # Analyze content structure from sitemap URLs
        content_structure = []
        seen_titles = set()
        
        for url in urls:
            if '/blog/' in url or '/fue/' in url:
                page_content = await extract_content(url)
                if page_content:
                    # Extract headings and key sections
                    headings = [h['text'] for h in page_content.get('headings', [])]
                    content_structure.append({
                        'url': url,
                        'title': page_content.get('title', ''),
                        'main_headings': [h for h in headings if h.lower().startswith('h1')],
                        'sub_headings': [h for h in headings if h.lower().startswith('h2')],
                        'word_count': len(page_content.get('text', '').split()),
                        'keywords': list(set(
                            [w.lower() for w in page_content.get('text', '').split() 
                             if w.lower() in ['fue', 'transplant', 'hairline', 'restoration']]
                        ))
                    })
                    seen_titles.add(page_content.get('title', ''))
        
        return {
            'domain': domain,
            'content_structure': content_structure,
            'unique_titles': list(seen_titles),
            'total_pages': len(content_structure)
        }
        
    except Exception as e:
        print(f"Sitemap analysis failed: {e}")
        return {}
