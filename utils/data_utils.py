import csv
from datetime import datetime
from typing import List, Dict

from models.venue import Venue


def is_duplicate_venue(venue_name: str, seen_names: set) -> bool:
    return venue_name in seen_names


def is_complete_venue(venue: dict, required_keys: list) -> bool:
    return all(key in venue for key in required_keys)


def save_venues_to_csv(venues: list, filename: str):
    if not venues:
        print("No venues to save.")
        return

    # Use field names from the Venue model
    fieldnames = Venue.model_fields.keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(venues)
    print(f"Saved {len(venues)} venues to '{filename}'.")


def save_content_to_csv(content: Dict, filename: str) -> str:
    """Save structured content to client-ready CSV"""
    fields = [
        'Section Type', 'Heading', 'Content', 
        'Primary Keywords', 'Secondary Keywords',
        'Word Count', 'SEO Score', 'Status'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for section in content.get('sections', []):
            writer.writerow({
                'Section Type': section.get('heading', 'H2'),
                'Heading': section.get('title', ''),
                'Content': section.get('content', ''),
                'Primary Keywords': ', '.join(content['keywords']['primary']),
                'Secondary Keywords': ', '.join(content['keywords']['secondary']),
                'Word Count': len(section.get('content', '').split()),
                'SEO Score': section.get('seo_score', 'Pending'),
                'Status': 'DRAFT'
            })
    
    return filename


def import_feedback_csv(filename: str) -> Dict:
    """Import client feedback from CSV"""
    # Implementation for processing feedback
    return {"status": "Feedback imported"}


def add_ai_insights_to_csv(content: Dict) -> Dict:
    """Enhance CSV data with AI-generated insights"""
    for section in content['sections']:
        # Add readability score
        section['readability'] = calculate_readability(section['content'])
        
        # Add keyword analysis
        section['keyword_density'] = {
            'primary': count_keywords(section['content'], content['keywords']['primary']),
            'secondary': count_keywords(section['content'], content['keywords']['secondary'])
        }
    
    return content


def calculate_readability(text: str) -> float:
    """Simple Flesch reading ease score"""
    # Implementation here
    return 70.0  # Placeholder


def count_keywords(text: str, keywords: List[str]) -> Dict:
    """Count keyword occurrences"""
    return {kw: text.lower().count(kw.lower()) for kw in keywords}
