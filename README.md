# Web Scraper with AI-Powered Analysis

A modern web scraping application that uses natural language processing to extract and analyze content from websites. Built with FastAPI, LangChain, Docker, and PostgreSQL.

## Features

- **Natural Language Query Processing**: Enter queries containing a topic in natural language (e.g., "Can you scrape the content from https://example.com?")
- **AI-Powered URL Extraction**: Uses GPT-3.5-turbo to intelligently extract URLs from user queries
- **Content Analysis**: Automatically generates summaries of scraped content using LangChain and OpenAI
- **Database Caching**: Stores scraped content for quick retrieval of previously scraped URLs
- **Modern UI**: Clean and responsive interface with real-time progress updates
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: LangChain, OpenAI GPT-3.5-turbo
- **Database**: PostgreSQL
- **Containerization**: Docker
- **Web Scraping**: BeautifulSoup4, aiohttp

## Getting Started

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

4. Access the application at `http://localhost:8000`

## Usage

1. Enter a query containing a URL in the text area
2. Click "Start Scraping"
3. Wait for the content to be scraped and analyzed
4. View the results, including:
   - Page title
   - Extracted content
   - AI-generated summary
   - Original URL
   - Timestamp

## Project Structure

```
app/
├── core/
│   └── scraper.py      # Web scraping and content extraction
├── db/
│   ├── database.py     # Database configuration
│   └── models.py       # Database models
├── static/            # Static files
├── templates/         # HTML templates
└── main.py           # FastAPI application
```

## Contributing

This is a work in progress. Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 