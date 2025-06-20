import json
from agents.news_event_agent import NewsEventAgent, NewsEventInput
from agents.sentiment_agent import SentimentAgent, SentimentInput
from agents.impact_agent import ImpactAgent, ImpactInput

import os
print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_BASE"))

def load_test_articles(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_outputs(results: list, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    articles = load_test_articles("data/test_articles.json")

    news_agent = NewsEventAgent()
    sentiment_agent = SentimentAgent()
    impact_agent = ImpactAgent()

    all_results = []

    for article in articles:
        print(f"Processing {article['article_id']}...")

        # Step 1: Extract news event info
        event_input = NewsEventInput(**article)
        event_output = news_agent.run(event_input)

        # Step 2: Analyze sentiment
        full_text = f"{article['headline']}\n\n{article['content']}"
        sentiment_input = SentimentInput(text=full_text)
        sentiment_output = sentiment_agent.run(sentiment_input)

        # Step 3: Predict impact
        impact_input = ImpactInput(
            tickers=event_output.tickers,
            events=event_output.events,
            sentiment=sentiment_output.sentiment,
            score=sentiment_output.score
        )
        impact_output = impact_agent.run(impact_input)

        result = {
            "article_id": article["article_id"],
            "tickers": event_output.tickers,
            "events": event_output.events,
            "sentiment": sentiment_output.sentiment,
            "sentiment_score": sentiment_output.score,
            "prediction": impact_output.prediction,
            "confidence": impact_output.confidence
        }

        all_results.append(result)

    save_outputs(all_results, "outputs/results.json")
    print("✅ All articles processed. Results saved to outputs/results.json.")

if __name__ == "__main__":
    main()
