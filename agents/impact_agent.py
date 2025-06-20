import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic_ai.agent import Agent
from pydantic import BaseModel, Field
import json
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)

class ImpactInput(BaseModel):
    tickers: list[str]
    events: list[str]
    sentiment: str
    score: float

class ImpactOutput(BaseModel):
    prediction: dict = Field(..., description="Dictionary of ticker → {direction, change_pct}")
    confidence: float = Field(..., description="Overall confidence in prediction")

class ImpactAgent(Agent[ImpactInput, ImpactOutput]):
    system_prompt = (
        "You are a financial analyst agent that uses sentiment and news events "
        "to predict short-term stock price movements."
    )

    def run(self, input: ImpactInput) -> ImpactOutput:
        prompt = (
            f"Based on the following:\n"
            f"- Tickers: {input.tickers}\n"
            f"- Events: {input.events}\n"
            f"- Sentiment: {input.sentiment} (score={input.score})\n\n"
            "Predict the market direction ('up', 'down', or 'neutral') and estimated percent change for each ticker.\n"
            "Also include an overall confidence score (0-100).\n\n"
            "Return your answer ONLY as a single valid JSON object inside a markdown ```json code block, like this:\n\n"
            "```json\n"
            "{\n"
            "  \"prediction\": {\n"
            "    \"TSLA\": {\n"
            "      \"direction\": \"up\",\n"
            "      \"change_pct\": 3.5\n"
            "    }\n"
            "  },\n"
            "  \"confidence\": 90\n"
            "}\n"
            "```\n\n"
            "❗ Do not include any text or explanation before or after the JSON code block."
        )


        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            stream=False,
        )
        return self.parse_response(response.choices[0].message.content, ImpactOutput)
    
    def parse_response(self, response_text: str, output_model):

        cleaned = re.sub(r"```(json)?", "", response_text).replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            return output_model(**parsed)
        except Exception as e:
            print("❌ Failed to parse cleaned response:")
            print(cleaned)
            raise e

