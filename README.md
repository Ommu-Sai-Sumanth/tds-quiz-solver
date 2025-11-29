# TDS Quiz Solver API

This project provides a FastAPI-based web service that uses Playwright to automatically open and solve quiz pages from the university TDS system.

The API exposes one main endpoint:

## POST /solve
Accepts:
- `url`: The quiz page URL
- `answer_type`: (optional) Type of answer field (text, mcq, checkbox, numerical)
- `model`: (optional) AI model to use

Example Request:
```json
{
  "url": "https://example.com/quiz",
  "answer_type": "text"
}
