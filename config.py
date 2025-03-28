# config.py

import os
from typing import List, Dict

# Base configuration
BASE_URL = "https://www.maximhairrestoration.com"
LOCATION_BASE = f"{BASE_URL}/locations"

# SEO Configuration
PRIMARY_LOCATION = "new-york"
PRIMARY_SERVICE = "fue-hair-transplant"
TARGET_URL = f"{LOCATION_BASE}/{PRIMARY_LOCATION}/{PRIMARY_SERVICE}"

# Competitor Analysis
COMPETITOR_DOMAINS = [
    'https://www.bosley.com',
    'https://www.restorehair.com',
    'https://www.nashvillehairdoctor.com',
    'https://www.hairphysician.com',
    'https://www.foundhair.com',
]

COMPETITOR_PATHS = {
    'bosley': '/procedures/fue-hair-transplant',
    'restorehair': '/treatments/fue-hair-transplant',
    'hairphysician': '/services/fue-transplant',
}

# Content Analysis Settings
CONTENT_REQUIREMENTS = {
    'min_word_count': 1500,
    'max_heading_depth': 3,
    'target_keyword_density': 1.5,
    'required_sections': [
        'overview',
        'benefits',
        'procedure',
        'recovery',
        'faq'
    ],
    'content_structure': {
        'intro_length': 200,
        'section_min_length': 300,
        'max_paragraph_length': 150
    }
}

# Add to existing config
COPY_STRATEGY = {
    'primary_keyword': "FUE Hair Transplants in NYC",
    'secondary_keyword_count': 5,
    'tone': "professional",
    'target_reader': "potential patients aged 30-50",
    'cta_phrases': [
        "Schedule Your Consultation",
        "Learn More About Our Techniques",
        "View Before/After Gallery"
    ]
}

PROMPT_TEMPLATES = {
    'new_page': """
    Create SEO-optimized content for a new {service} page in {location}. 
    Use primary keyword 3 times naturally. Include:
    - 500-word overview
    - 3 key benefits
    - Procedure steps
    - Recovery timeline
    - 5 FAQs
    """,
    
    'improve_page': """
    Improve existing content for {url}. Focus on:
    - Better keyword integration
    - Content structure
    - Readability
    - Add missing sections: {missing_sections}
    Keep core content but enhance flow and SEO value.
    """
}

AI_AGENTS = {
    'researcher': {
        'prompt': """
        Analyze these content sources:
        {sources}
        Key requirements:
        - Primary keyword: {primary_keyword}
        - Secondary keywords: {secondary_keywords}
        - Tone: {tone}
        Output: Content outline with 5-7 main sections
        """
    },
    'writer': {
        'prompt': """
        Write detailed content for section: {section_title}
        Guidelines:
        - Use {primary_keyword} 2-3 times naturally
        - Include 1-2 secondary keywords
        - Length: {min_words}-{max_words} words
        - Tone: {tone}
        """
    },
    'seo_optimizer': {
        'prompt': """
        Optimize this content for SEO:
        {content}
        Required improvements:
        - Better keyword integration
        - Add internal links
        - Improve readability
        - Ensure proper heading structure
        """
    },
    'editor': {
        'prompt': """
        Final polish for content:
        {content}
        Check:
        - Grammar/spelling
        - Consistency
        - Readability score > 80
        - Remove jargon
        """
    }
}
