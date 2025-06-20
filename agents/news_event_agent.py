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

class NewsEventInput(BaseModel):
    article_id: str
    headline: str
    content: str

class NewsEventOutput(BaseModel):
    tickers: list[str] = Field(..., description="Detected stock symbols")
    events: list[str] = Field(..., description="Detected event types")

class NewsEventAgent(Agent[NewsEventInput, NewsEventOutput]):
    system_prompt = "You are a financial news parser. Extract tickers and event types."

    def run(self, input: NewsEventInput) -> NewsEventOutput:
        prompt = (
            f"You are a financial news parser.\n"
            f"Given the following article:\n"
            f"- Headline: {input.headline}\n"
            f"- Content: {input.content}\n\n"
            "Extract the list of stock tickers mentioned (by symbol only), and a list of event types.\n"
            "Return a single valid JSON object with this structure inside a markdown ```json code block:\n\n"
            "```json\n"
            "{\n"
            "  \"tickers\": [\"TSLA\"],\n"
            "  \"events\": [\"earnings_report\", \"regulatory_concern\"]\n"
            "}\n"
            "```\n\n"
            "Instructions:\n"
            "- 'tickers' should be a list of stock ticker symbols (like TSLA, AMZN).\n"
            "- 'events' should be a list of simple strings like 'earnings_report', 'merger', 'regulatory_concern'.\n"
            "- Do not include reasoning or extra explanations.\n"
            "- Only return the code block."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,
            stream=False,
        )
        return self.parse_response(response.choices[0].message.content, NewsEventOutput)

    def parse_response(self, response_text: str, output_model):
        

        cleaned = re.sub(r"```(json)?", "", response_text).replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            return output_model(**parsed)
        except Exception as e:
            print("❌ Failed to parse cleaned response:")
            print(cleaned)
            raise e
