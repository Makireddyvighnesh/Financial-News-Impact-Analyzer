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

class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str = Field(..., description="Positive, Neutral, or Negative")
    score: float = Field(..., description="Confidence between 0 and 1")

class SentimentAgent(Agent[SentimentInput, SentimentOutput]):
    system_prompt = "You are a financial sentiment analyst. Provide sentiment and confidence."

    def run(self, input: SentimentInput) -> SentimentOutput:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Analyze the sentiment of this financial text:\n\n"
                    f"{input.text}\n\n"
                    "Return your response in this exact JSON format inside a markdown ```json code block:\n\n"
                    "```json\n"
                    "{\n"
                    "  \"sentiment\": \"Positive\",\n"
                    "  \"score\": 0.92\n"
                    "}\n"
                    "```\n"
                    "Do not include any explanation outside the JSON block."

                ),
            },
        ]
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,
            stream=False,
        )
        return self.parse_response(resp.choices[0].message.content, SentimentOutput)
    
    def parse_response(self, response_text: str, output_model):
        try:
            match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = re.search(r"\{.*\}", response_text, re.DOTALL).group(0)

            parsed = json.loads(cleaned)
            return output_model(**parsed)

        except Exception as e:
            print("❌ Failed to parse cleaned response:")
            print(response_text)
            raise e


