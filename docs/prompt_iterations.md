# Prompt Iterations Log

## NewsEventAgent
### v1
"Extract stock tickers and event types from the article."
— Too generic; missed non-US stocks.

### v2
"You are a financial parser. Extract all stock tickers (e.g., TSLA, BABA) and event types (earnings, M&A, FDA approval)."

---

## SentimentAgent
### v1
"Analyze the sentiment of this financial text."
— Worked well, but sometimes returned generic responses.

### v2
"Rate the sentiment as Positive, Neutral, or Negative, and give a confidence score between 0 and 1."

---

## ImpactAgent
### v1
"Predict the direction and percentage change for tickers."
— Returned incomplete JSON.

### v2
"Given tickers, events, and sentiment score, predict direction (up/down/neutral) and estimated % change. Return as JSON."
