from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict

class ContentAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        
        self.comparison_prompt = PromptTemplate(
            input_variables=["our_content", "competitor_content", "keywords"],
            template="""Analyze our hair transplant content vs competitors:
            
            Our Content:
            {our_content}
            
            Competitor Content:
            {competitor_content}
            
            Target Keywords:
            {keywords}
            
            Identify 3 areas where competitors are better and 2 unique selling points we should emphasize.
            Provide analysis in bullet points."""
        )
        
        self.optimization_prompt = PromptTemplate(
            input_variables=["original", "analysis", "keywords"],
            template="""Optimize hair transplant page content:
            
            Original Content:
            {original}
            
            SEO Analysis:
            {analysis}
            
            Target Keywords (use secondary 3x):
            {keywords}
            
            Generate improved copy with proper keyword distribution and compelling CTAs."""
        )

    def compare_content(self, our_content: str, competitors: List[Dict], keywords: List[str]) -> str:
        chain = LLMChain(llm=self.llm, prompt=self.comparison_prompt)
        return chain.run({
            'our_content': our_content,
            'competitor_content': '\n\n'.join([c['text'] for c in competitors]),
            'keywords': ', '.join(keywords)
        })
    
    def generate_optimized_copy(self, original: str, analysis: str, keywords: List[str]) -> str:
        chain = LLMChain(llm=self.llm, prompt=self.optimization_prompt)
        return chain.run({
            'original': original,
            'analysis': analysis,
            'keywords': ', '.join(keywords)
        }) 