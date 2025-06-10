# Web Scraper with AI-Powered Analysis

A Wikipedia scraping application that uses natural language processing to extract and analyze content from Wikipedia articles. Built with FastAPI, LangChain, Docker, and PostgreSQL.

## Features

- **Natural Language Query Processing**: Enter queries in natural language (e.g., "Tell me about artificial intelligence" or "What is quantum computing?")
- **Topic Extraction**: Uses GPT-3.5-turbo to extract Wikipedia topics from user queries
- **Content Analysis**: Generates concise summaries (300 words) of Wikipedia articles using LangChain and OpenAI
- **Database Caching**: Stores scraped content for quick retrieval of previously searched topics
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Health Monitoring**: Built-in health check endpoint for monitoring application status

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

1. Enter a topic you'd like to learn about in the text area
2. Click "Search Wikipedia"
3. Wait for the content to be scraped and analyzed
4. View the results, including:
   - Article title with link to Wikipedia
   - AI-generated summary (limited to 300 words)
   - Full article content
   - Last updated timestamp

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