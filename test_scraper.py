import requests
import time
from typing import List
from urllib.parse import urlparse
import json

def validate_urls(urls: List[str]) -> List[str]:
    """Validate URLs and return only valid ones."""
    valid_urls = []
    for url in urls:
        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                valid_urls.append(url)
            else:
                print(f"Invalid URL: {url}")
        except Exception as e:
            print(f"Error validating URL {url}: {str(e)}")
    return valid_urls

def scrape_urls(urls: List[str]) -> dict:
    """Send URLs to the scraper API and monitor the job."""
    # Validate URLs first
    valid_urls = validate_urls(urls)
    if not valid_urls:
        return {"error": "No valid URLs provided"}

    # Start scraping job
    response = requests.post(
        "http://localhost:8000/api/scrape",
        json={"urls": valid_urls}
    )
    
    if response.status_code != 200:
        return {"error": f"Failed to start scraping job: {response.text}"}
    
    job_data = response.json()
    job_id = job_data["job_id"]
    
    # Monitor job status
    while True:
        status_response = requests.get(f"http://localhost:8000/api/jobs/{job_id}")
        if status_response.status_code != 200:
            return {"error": f"Failed to get job status: {status_response.text}"}
        
        job_status = status_response.json()
        if job_status["status"] in ["completed", "failed"]:
            return job_status
        
        print(f"Job status: {job_status['status']}")
        time.sleep(2)  # Wait 2 seconds before checking again

def main():
    # Example URLs to scrape
    test_urls = [
        "invalid-url",  # This will be filtered out
        "https://github.com"
    ]
    
    print("Starting scraping job...")
    result = scrape_urls(test_urls)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nScraping completed!")
        print("\nResults:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 