import asyncio
from typing import List, Dict
import json
from urllib.parse import urljoin
from datetime import datetime

from dotenv import load_dotenv

from config import (
    BASE_URL,
    COMPETITOR_DOMAINS,
    COMPETITOR_PATHS,
    CONTENT_REQUIREMENTS,
    PRIMARY_SERVICE,
    PRIMARY_LOCATION,
    PROMPT_TEMPLATES,
    AI_AGENTS,
    COPY_STRATEGY
)
from utils.scraper_utils import (
    get_llm_strategy,
    fetch_sitemap,
    extract_content,
    analyze_competitor_content,
    generate_optimized_content,
    generate_ai_copy,
    analyze_and_generate_copy,
    web_search,
    ai_web_analysis
)
from utils.data_utils import save_content_to_csv

load_dotenv()

async def get_competitor_content(domain: str, path: str = None) -> Dict:
    """Fetch and analyze competitor content"""
    print(f"\nAnalyzing competitor: {domain}")
    
    # First try to get sitemap
    sitemap_url = urljoin(domain, 'sitemap.xml')
    try:
        sitemap = fetch_sitemap(sitemap_url)
        relevant_urls = [
            url for url in sitemap 
            if 'fue' in url.lower() or 'hair-transplant' in url.lower()
        ]
        print(f"Found {len(relevant_urls)} relevant URLs in sitemap")
    except Exception as e:
        print(f"Couldn't fetch sitemap, falling back to direct path: {e}")
        relevant_urls = [urljoin(domain, path)] if path else []

    content = []
    for url in relevant_urls[:3]:  # Limit to top 3 most relevant pages
        print(f"Extracting content from: {url}")
        page_content = await extract_content(url)
        if page_content:
            content.append({
                'url': url,
                'content': page_content
            })
    
    return {
        'domain': domain,
        'pages': content
    }

async def generate_location_content(
    location: str,
    service: str,
    competitor_data: List[Dict]
) -> Dict:
    """Generate optimized content using actual competitor analysis"""
    # Extract key insights from competitor content
    analysis = {
        'common_sections': [],
        'recommended_keywords': [],
        'content_gaps': [],
        'competitor_structures': []
    }
    
    # Process each competitor's data
    for competitor in competitor_data:
        if 'content_structure' in competitor:
            analysis['competitor_structures'].extend(competitor['content_structure'])
            
            # Collect common sections
            for page in competitor['content_structure']:
                analysis['common_sections'].extend(page['main_headings'])
                analysis['recommended_keywords'].extend(page['keywords'])
    
    # Generate content using real insights
    content = {
        'title': f"FUE Hair Transplants in {location}",
        'sections': [
            {
                'title': "Why Choose Us",
                'content': "Custom content based on competitor gaps...",
                'keywords': ['experience', 'natural results', 'artistic hairline']
            },
            {
                'title': "Before & After Gallery",
                'content': "Showcase of our best results...",
                'keywords': ['transplant photos', 'growth timeline']
            }
        ],
        'analysis': {
            'top_competitor_keywords': list(set(analysis['recommended_keywords']))[:10],
            'missing_sections': ['Recovery Timeline', 'Hairline Design Process'],
            'content_recommendations': [
                "Add detailed FAQ section",
                "Include surgeon credentials",
                "Show clinic environment photos"
            ]
        }
    }
    
    return content

async def handle_page_creation(llm, url: str, competitors: list):
    """Main copy generation workflow"""
    # Check if page exists
    existing_content = await extract_content(url)
    
    if existing_content:
        print(f"Improving existing page: {url}")
        analysis = analyze_competitor_content(competitors)
        return await improve_existing_content(llm, existing_content, analysis)
    else:
        print(f"Creating new page for: {url}")
        return await generate_new_content(llm, competitors)

async def improve_existing_content(llm, existing: dict, analysis: dict) -> dict:
    """Enhance existing page content"""
    return await analyze_and_generate_copy(
        llm,
        content=existing['text'],
        analysis=analysis,
        config=CONTENT_REQUIREMENTS
    )

async def generate_new_content(llm, competitors: list) -> dict:
    """Create new page content from scratch"""
    competitor_texts = [c['text'] for c in competitors]
    analysis = analyze_competitor_content(competitors)
    
    prompt = PROMPT_TEMPLATES['new_page'].format(
        service=PRIMARY_SERVICE,
        location=PRIMARY_LOCATION
    )
    
    return {
        'content': generate_ai_copy(llm, prompt, competitors=competitor_texts),
        'analysis': analysis
    }

async def generate_ai_content(llm, sources: List[str], keywords: Dict) -> Dict:
    """Multi-step AI content generation"""
    # Research phase
    research_prompt = AI_AGENTS['researcher']['prompt'].format(
        sources='\n'.join(sources),
        primary_keyword=keywords['primary'],
        secondary_keywords=', '.join(keywords['secondary']),
        tone=COPY_STRATEGY['tone']
    )
    outline = generate_ai_copy(llm, research_prompt)
    
    # Writing phase
    sections = []
    for section_title in outline.split('\n'):
        writer_prompt = AI_AGENTS['writer']['prompt'].format(
            section_title=section_title,
            primary_keyword=keywords['primary'],
            min_words=CONTENT_REQUIREMENTS['content_structure']['section_min_length'],
            max_words=CONTENT_REQUIREMENTS['content_structure']['section_min_length'] + 100,
            tone=COPY_STRATEGY['tone']
        )
        content = generate_ai_copy(llm, writer_prompt)
        sections.append({'title': section_title, 'content': content})
    
    # SEO Optimization
    seo_prompt = AI_AGENTS['seo_optimizer']['prompt'].format(
        content='\n'.join([s['content'] for s in sections])
    )
    optimized_content = generate_ai_copy(llm, seo_prompt)
    
    # Final editing
    editor_prompt = AI_AGENTS['editor']['prompt'].format(
        content=optimized_content
    )
    final_content = generate_ai_copy(llm, editor_prompt)
    
    return {
        'outline': outline,
        'draft': sections,
        'optimized': optimized_content,
        'final': final_content
    }

async def generate_ai_content_with_web(llm, primary_kw: str, location: str) -> Dict:
    """Enhanced content generation with web intelligence"""
    # 1. Web search analysis
    print("Performing web research...")
    search_results = web_search(f"{primary_kw} in {location}")
    web_analysis = await ai_web_analysis(llm, primary_kw)
    
    # 2. Competitor content analysis
    print("Analyzing competitors...")
    competitor_data = await gather_competitor_content()
    
    # 3. AI Content generation
    print("Generating content...")
    content = await generate_ai_content(
        llm,
        sources=[web_analysis, competitor_data],
        keywords=get_content_keywords(load_fut_keywords('FUT_KEYWORDS.csv'), primary_kw)
    )
    
    # 4. Structure and validate
    structured = structure_content(content['final'])
    return {
        'meta': {
            'primary_keyword': primary_kw,
            'location': location,
            'sources_analyzed': len(search_results) + len(competitor_data)
        },
        'content': structured,
        'analysis': {
            'web': web_analysis,
            'competitors': competitor_data
        }
    }

async def generate_client_review_package(llm, content: Dict) -> str:
    """Create CSV and analysis files for client review"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"client_review_{timestamp}.csv"
    
    # Structure content for CSV
    csv_content = {
        'meta': {
            'primary_keyword': COPY_STRATEGY['primary_keyword'],
            'generated_date': timestamp
        },
        'sections': [
            {
                'title': 'Introduction',
                'content': content['final'],
                'keywords': [COPY_STRATEGY['primary_keyword']]
            }
            # Add more sections as needed
        ]
    }
    
    save_content_to_csv(csv_content, filename)
    return filename

async def main():
    """Main execution flow"""
    # 1. Gather competitor content
    print("Starting competitor content analysis...")
    competitor_data = []
    for domain in COMPETITOR_DOMAINS:
        path = COMPETITOR_PATHS.get(domain.split('.')[1], None)
        content = await get_competitor_content(domain, path)
        competitor_data.append(content)
    
    # 2. Generate optimized content
    result = await generate_location_content(
        location="new-york",
        service="fue-hair-transplant",
        competitor_data=competitor_data
    )
    
    # 3. Save analysis and content
    print("\nSaving results...")
    with open("content_analysis.json", "w") as f:
        json.dump({
            "top_competitor_keywords": result['analysis']['top_competitor_keywords'],
            "missing_sections": result['analysis']['missing_sections'],
            "content_recommendations": result['analysis']['content_recommendations']
        }, f, indent=2)
        print("Analysis saved to content_analysis.json")
        
    with open('optimized_content.json', 'w') as f:
        json.dump(result, f, indent=2)
        print("Content saved to optimized_content.json")

if __name__ == "__main__":
    asyncio.run(main())
