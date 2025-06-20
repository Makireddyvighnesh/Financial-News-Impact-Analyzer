# agents/evaluation_agent.py

from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)

class EvaluationInput(BaseModel):
    headline: str
    content: str
    prediction_json: str

class EvaluationOutput(BaseModel):
    plausible: bool = Field(..., description="Is the prediction reasonable?")
    score: float = Field(..., description="Confidence in reasonableness, 0-1")
    rationale: str = Field(..., description="Explanation")

class EvaluationAgent(Agent[EvaluationInput, EvaluationOutput]):
    system_prompt = "You are a financial evaluator who verifies the plausibility of AI-generated market predictions based on financial news."

    def run(self, input: EvaluationInput) -> EvaluationOutput:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"News Headline: {input.headline}\n"
                f"Content: {input.content}\n\n"
                f"Prediction (JSON):\n```json\n{input.prediction_json}\n```\n\n"
                "Does this prediction make sense based on the article?\n"
                "Return only valid JSON in the following format:\n"
                "```json\n"
                "{\n"
                "  \"plausible\": true,\n"
                "  \"score\": 0.85,\n"
                "  \"rationale\": \"Explanation here\"\n"
                "}\n"
                "```"
            )}
        ]

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            stream=False,
        )

        import json, re
        try:
            text = resp.choices[0].message.content
            cleaned = re.search(r"```json(.*?)```", text, re.DOTALL).group(1).strip()
            parsed = json.loads(cleaned)
            return EvaluationOutput(**parsed)
        except Exception as e:
            print("❌ Failed to parse evaluation response:")
            print(text)
            raise e
