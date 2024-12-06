# src/ChatwithCSV.py

import asyncio
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.gemini import Gemini
from .logging_config import get_logger  # Ensure correct relative import

class ChatwithCSV:
    def __init__(self, api_key: str, df: pd.DataFrame, provider: str = "GEMINI") -> None:
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.provider = provider.upper()
        self.df = df
        self.instruction = (
            "You are an excellent data analyst who can answer questions based on a given pandas dataframe. "
            "If you cannot figure out the answer, just politely say `The given context does not provide answer to the following problem`."
        )
        self.logger.debug(f"Initializing ChatwithCSV with provider: {self.provider}")

        if self.provider == "OPENAI":
            self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-4o-mini",
                openai_api_key=self.api_key,
                streaming=True
            )
            self.agent_executor = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                agent_type="tool-calling",
                verbose=True,
                allow_dangerous_code=True,
                prefix=self.instruction,
            )
            self.logger.debug("Initialized OpenAI agent executor.")
        elif self.provider == "GEMINI":
            self.llm = Gemini(
                model="models/gemini-1.5-flash",
                api_key=self.api_key,  # Assumes GOOGLE_API_KEY env var by default
            )
            self.agent_executor = PandasQueryEngine(
                df=self.df,
                verbose=True,
                llm=self.llm
            )
            self.logger.debug("Initialized Gemini agent executor.")
        else:
            self.logger.error(f"Unsupported provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def chat_with_a_df(self, question: str) -> str:
        self.logger.info(f"Received question: {question}")
        loop = asyncio.get_event_loop()
        try:
            if self.provider == "OPENAI":
                # Await the async invoke and then get the output
                result = await self.agent_executor.ainvoke({"input": question})
                ans = result.get("output", "I don't know")
            else:
                # Await the run_in_executor call
                ans = await loop.run_in_executor(None, self.agent_executor.query, question)
                # Assuming the Gemini query_engine.query returns a string directly
                ans = ans.__str__()
            self.logger.debug(f"Response: {ans}")
            return ans
        except Exception as e:
            self.logger.error(f"Error in chat_with_a_df: {e}")
            return "I'm sorry, I couldn't process your request."


if __name__ == "__main__":
    import os
    import json
    import sys

    async def main():
        # Ensure API keys are set
        openai_key = os.getenv("OPENAI_API_KEY", "")
        google_api_key = os.getenv("GOOGLE_API_KEY", "")

        # Initialize ChatwithCSV with Gemini by default
        if not google_api_key:
            logger.error("GOOGLE_API_KEY is not set in environment variables.")
            sys.exit(1)

        # Load DataFrame
        csv_path = r"src\data\vitals.csv"  # Update this path as needed
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV file from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            sys.exit(1)

        # Initialize ChatwithCSV
        chat = ChatwithCSV(api_key=google_api_key, df=df, provider="GEMINI")

        samples = []
        while True:
            question = input("Please enter your question here (type 'exit' to quit): ")
            if question.lower() == "exit":
                break
            ans = await chat.chat_with_a_df(question=question)
            print(ans)
            samples.append({"question": question, "answer": ans})

        # Save samples to JSON
        try:
            with open("sample_queries.json", "w") as f:
                json.dump(samples, f, indent=4)
            logger.info("Sample queries saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save sample queries: {e}")

        print("Files Saved")

    # Initialize logger for standalone script
    from .logging_config import get_logger
    logger = get_logger(__name__)

    asyncio.run(main())
