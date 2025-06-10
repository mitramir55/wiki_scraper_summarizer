from typing import Optional, Dict, Any
import asyncio
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import json
from langchain.prompts import PromptTemplate

class AsyncWebScraper:
    def __init__(
        self,
        timeout: int = 15,
        max_retries: int = 2
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=0
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_page(self, url: str) -> str:
        """Fetch a single page with retries."""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, timeout=self.timeout) as response:
                    if response.status == 404:
                        raise Exception(f"404, message='Not Found', url=URL('{url}')")
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)

    def extract_content(self, soup: BeautifulSoup) -> str:
        """Extract content from the page."""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'meta']):
            element.decompose()

        # Get all text elements in their natural order from the entire page
        results_text = []
        
        # Find all text elements in order
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'blockquote']):
            text = element.get_text(separator=' ', strip=True)
            if text:  # Only add non-empty text
                results_text.append(text)
        
        # Join with newlines to preserve structure
        content = '\n\n'.join(results_text)
        
        return content

    async def generate_summary(self, text: str) -> str:
        """Generate a summary using LangChain and OpenAI."""
        # Split text into smaller chunks
        docs = [Document(page_content=text)]
        split_docs = self.text_splitter.split_documents(docs)
        
        # Create a proper prompt template
        summary_template = """Write a concise summary of the following text. The summary should:
        1. Be at most 300 words
        2. Capture the main points and key information
        3. Be clear and easy to understand
        4. Maintain the most important details
        5. Be written in a neutral, informative tone

        Text to summarize:
        {text}

        Summary:"""
        
        prompt = PromptTemplate(
            template=summary_template,
            input_variables=["text"]
        )
        
        # Use a custom chain with the prompt
        chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            verbose=False,
            prompt=prompt
        )
        
        # Generate summary
        summary = await chain.arun(split_docs)
        
        # Ensure summary is not longer than 300 words
        words = summary.split()
        if len(words) > 300:
            summary = ' '.join(words[:300]) + '...'
            
        return summary

    def create_error_result(self, url: str, error: Exception) -> Dict[str, Any]:
        """Create a properly formatted error result."""
        error_msg = str(error)
        
        # Handle 404 errors specifically
        if "404" in error_msg and "Not Found" in error_msg:
            # Extract the topic from the URL
            topic = url.split("/wiki/")[-1].replace("_", " ")
            return {
                'url': url,
                'error': f"Article not found: '{topic}'. Please check the spelling or try a different topic.",
                'error_type': 'not_found',
                'timestamp': datetime.utcnow().isoformat(),
                'title': '',
                'text': '',
                'summary': ''
            }
        
        return {
            'url': url,
            'error': error_msg,
            'error_type': 'general_error',
            'timestamp': datetime.utcnow().isoformat(),
            'title': '',
            'text': '',
            'summary': f"Error generating summary: {error_msg}"
        }

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL and process its content."""
        try:
            html = await self.fetch_page(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract content
            text = self.extract_content(soup)
            
            # Generate AI summary
            summary = await self.generate_summary(text)
            
            content = {
                'url': url,
                'timestamp': datetime.utcnow().isoformat(),
                'title': soup.title.string if soup.title else '',
                'text': text,
                'summary': summary
            }
            
            # Log success without full content
            print(f"Successfully scraped article: {content['title']}")
            
            # Ensure all content is JSON serializable
            return json.loads(json.dumps(content))
        except Exception as e:
            error_result = self.create_error_result(url, e)
            print(f"Error scraping {url}: {error_result['error']}")
            return error_result
