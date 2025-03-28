class PageTemplateGenerator:
    def __init__(self, analysis_data: dict):
        self.data = analysis_data
    
    def generate_landing_page(self) -> str:
        return f"""
        <!-- Generated Landing Page Template -->
        <section class="hero">
            <h1>{self.data['primary_keyword']} Services</h1>
            {self._generate_ctas('hero')}
        </section>
        
        <section class="features">
            {self._generate_feature_blocks()}
        </section>
        """
    
    def _generate_ctas(self, section_type: str) -> str:
        return '\n'.join(
            f'<a href="{cta["link"]}" class="{section_type}-cta">{cta["text"]}</a>'
            for cta in self.data.get('ctas', [])
            if cta['section'] == section_type
        )
    
    def _generate_feature_blocks(self) -> str:
        return '\n'.join(
            f'<div class="feature"><h3>{feat["title"]}</h3><p>{feat["content"]}</p></div>'
            for feat in self.data.get('content_flow', [])
            if feat['type'] == 'feature'
        )

    def generate_location_page(self, primary_kw: str, secondary_kws: list, content: str) -> str:
        return f'''
        <!-- wp:group {{"className":"seo-content"}} -->
        <div class="wp-block-group seo-content">
            <!-- wp:heading {{"level":1}} -->
            <h1>{primary_kw}</h1>
            <!-- /wp:heading -->
            
            <!-- wp:paragraph -->
            <p>{content}</p>
            <!-- /wp:paragraph -->
            
            <!-- wp:columns -->
            <div class="wp-block-columns">
                <!-- wp:column -->
                <div class="wp-block-column">
                    <!-- wp:heading {{"level":2}} -->
                    <h2>Why Choose Us for {primary_kw}</h2>
                    <!-- /wp:heading -->
                    <!-- wp:list -->
                    <ul>{"".join([f'<li>{kw}</li>' for kw in secondary_kws[:3]])}</ul>
                    <!-- /wp:list -->
                </div>
                <!-- /wp:column -->
            </div>
            <!-- /wp:columns -->
        </div>
        <!-- /wp:group -->
        
        <!-- wp:group {{"className":"schema-markup"}} -->
        <script type="application/ld+json">
        {{
            "@context": "https://schema.org",
            "@type": "MedicalClinic",
            "name": "{primary_kw}",
            "keywords": "{", ".join(secondary_kws)}"
        }}
        </script>
        <!-- /wp:group -->
        '''

    # Creates optimized pages with:
    # - SEO-friendly structure
    # - Schema markup
    # - Proper keyword distribution
    # - Location-specific content 