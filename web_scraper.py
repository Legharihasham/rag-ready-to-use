import requests
from bs4 import BeautifulSoup
import os
import time
import re
import logging
from urllib.parse import urlparse, urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_scraping.log"),
        logging.StreamHandler()
    ]
)

class WebScraper:
    def __init__(self, max_threads=5, delay=1.0):
        """
        Initialize the web scraper
        
        Args:
            max_threads: Maximum number of concurrent threads for scraping
            delay: Delay between requests to the same domain (in seconds)
        """
        self.max_threads = max_threads
        self.delay = delay
        self.visited_urls = set()
        self.last_request_time = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _respect_robots_txt(self, url):
        """
        Check if the URL is allowed by robots.txt
        Basic implementation - could be expanded
        
        Returns True if allowed, False if not
        """
        # This is a simplified check
        return True
    
    def _get_domain(self, url):
        """Extract domain from URL"""
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def _should_delay_request(self, url):
        """Check if we need to delay the request to respect rate limiting"""
        domain = self._get_domain(url)
        current_time = time.time()
        
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
        
        self.last_request_time[domain] = time.time()
    
    def _clean_text(self, text):
        """Clean and normalize extracted text"""
        # Replace multiple whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove extra newlines
        text = re.sub(r'\n+', '\n', text)
        # Trim whitespace
        return text.strip()
    
    def _extract_text_from_html(self, html_content, url):
        """
        Extract clean text content from HTML
        
        Args:
            html_content: Raw HTML content
            url: Source URL for reference
            
        Returns:
            Clean text extracted from HTML
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
                script_or_style.decompose()
            
            # Extract text from main content areas
            main_content = ""
            
            # Try to find main content area
            content_elements = soup.select("main, article, #content, .content, #main, .main, .post, .entry, .page, .article")
            
            if content_elements:
                # Use content from identified content areas
                for element in content_elements:
                    main_content += element.get_text(separator=' ', strip=True) + "\n\n"
            else:
                # If no content areas found, use the body
                main_content = soup.body.get_text(separator=' ', strip=True)
            
            # Try to get the title
            title = ""
            if soup.title:
                title = soup.title.string
            
            # Combine with title and clean
            full_text = f"Title: {title}\nURL: {url}\n\n{main_content}"
            return self._clean_text(full_text)
            
        except Exception as e:
            logging.error(f"Error extracting text from {url}: {e}")
            return f"Error processing {url}: {str(e)}"
    
    def scrape_url(self, url):
        """
        Scrape content from a single URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with url and extracted text
        """
        # Skip if already visited
        if url in self.visited_urls:
            return None
        
        # Mark as visited
        self.visited_urls.add(url)
        
        # Check robots.txt
        if not self._respect_robots_txt(url):
            logging.info(f"Skipping {url} - disallowed by robots.txt")
            return None
        
        # Respect rate limiting
        self._should_delay_request(url)
        
        try:
            logging.info(f"Scraping {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                text_content = self._extract_text_from_html(response.text, url)
                return {
                    "url": url,
                    "text": text_content,
                    "status": "success"
                }
            else:
                logging.warning(f"Failed to retrieve {url}: HTTP {response.status_code}")
                return {
                    "url": url,
                    "text": f"Failed to retrieve content: HTTP {response.status_code}",
                    "status": "error"
                }
                
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "text": f"Error scraping content: {str(e)}",
                "status": "error"
            }
    
    def scrape_urls_from_file(self, file_path):
        """
        Scrape content from URLs listed in a file
        
        Args:
            file_path: Path to file containing URLs (one per line)
            
        Returns:
            List of dictionaries with url and extracted text
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []
        
        # Read URLs from file
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]
        
        logging.info(f"Found {len(urls)} URLs in {file_path}")
        
        # Remove duplicates
        unique_urls = list(set(urls))
        
        # Scrape URLs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self.scrape_url, url): url for url in unique_urls}
            
            for future in futures:
                result = future.result()
                if result and result["status"] == "success":
                    results.append(result)
        
        logging.info(f"Successfully scraped {len(results)} out of {len(unique_urls)} URLs")
        return results
    
    def split_into_chunks(self, scraped_data, chunk_size=600, chunk_overlap=200):
        """
        Split scraped text into chunks for embedding
        
        Args:
            scraped_data: List of dictionaries with url and text
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of dictionaries with text chunks and metadata
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        all_chunks = []
        
        for item in scraped_data:
            if item["status"] == "success":
                chunks = text_splitter.split_text(item["text"])
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": item["url"],
                            "chunk_id": i,
                            "type": "web"
                        }
                    })
        
        return all_chunks


def main(links_file="Data/Links.txt", chunk_size=600, chunk_overlap=200):
    """
    Process web data from links file
    
    Args:
        links_file: Path to file containing URLs
        chunk_size: Size of chunks for processing
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunks with metadata
    """
    # Initialize scraper
    scraper = WebScraper(max_threads=5, delay=1.5)
    
    # Scrape URLs
    scraped_data = scraper.scrape_urls_from_file(links_file)
    
    # Split into chunks
    web_chunks = scraper.split_into_chunks(scraped_data, chunk_size, chunk_overlap)
    
    print(f"Created {len(web_chunks)} chunks from web data")
    return web_chunks


if __name__ == "__main__":
    main() 