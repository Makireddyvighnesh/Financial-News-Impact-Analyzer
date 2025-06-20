from agents.evaluation_agent import EvaluationAgent, EvaluationInput
import json

def load_predictions(path):
    with open(path, "r") as f:
        return json.load(f)

def load_ground_truth():
    return {
        "FIN-001": {"TSLA": "up"},
        "FIN-002": {"CURE": "up"},
        "FIN-003": {"AMZN": "neutral"},
        "FIN-004": {"FSB": "up"},
        "FIN-005": {"ByteDance": "neutral"}
    }

def load_articles(filepath: str):
    with open(filepath, 'r') as f:
        articles = json.load(f)
    # Convert list to dict keyed by article_id
    return {article["article_id"]: article for article in articles}

def main():
    preds = load_predictions("outputs/results.json")
    ground_truth = load_ground_truth()
    articles = load_articles("data/test_articles.json")
    eval_agent = EvaluationAgent()

    total, correct, confidence_sum, eval_score_sum = 0, 0, 0.0, 0.0

    print("Prediction Results:\n")

    for p in preds:
        article_id = p["article_id"]
        pred = p["prediction"]
        conf = p["confidence"]
        confidence_sum += conf

        print(f"Article ID: {article_id}")
        for ticker, info in pred.items():
            pred_dir = info["direction"]
            expected_dir = ground_truth.get(article_id, {}).get(ticker, "N/A")
            print(ground_truth.get(article_id,{}))

            print(f"Ticker: {ticker}")
            print(f"→ Predicted Direction: {pred_dir}")
            print(f"→ Expected Direction: {expected_dir}")
            print(f"→ % Change: {info['change_pct']}%")
            print(f"→ Model Confidence: {conf:.2f}")

            if pred_dir == expected_dir:
                correct += 1
            total += 1

            # Evaluate prediction quality using LLM
            article = articles.get(article_id)
            if article:
                eval_input = EvaluationInput(
                    headline=article["headline"],
                    content=article["content"],
                    prediction_json=json.dumps(p["prediction"], indent=2)
                )
                eval_result = eval_agent.run(eval_input)
                print(f"LLM Evaluation Score: {eval_result.score}")
                print(f"Rationale: {eval_result.rationale}")
                eval_score_sum += eval_result.score

        print("-" * 40)

    print(f"\nDirection Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Avg Model Confidence: {confidence_sum/len(preds):.2f}")
    print(f"Avg LLM Evaluation Score: {eval_score_sum/total:.2f}")

if __name__ == "__main__":
    main()
