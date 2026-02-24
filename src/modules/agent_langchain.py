# src/ChatwithCSV.py

import asyncio
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI
from .logging_config import get_logger  # Ensure correct relative import

class ChatwithCSV:
    def __init__(self, api_key: str, df: pd.DataFrame, agent: str = "Langchain") -> None:
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.df = df
        self.agent = agent
        self.openai_model = "gpt-4o-mini"
        self.instruction = (
            "You are an excellent data analyst who can answer questions based on a given pandas dataframe. "
            "If you cannot figure out the answer, just politely say `The given context does not provide answer to the following problem`."
        )
        self.logger.debug(f"Initializing ChatwithCSV with OpenAI and agent: {self.agent}")

        if self.agent == "Langchain":
            self.llm = ChatOpenAI(
                temperature=0,
                model=self.openai_model,
                openai_api_key=self.api_key,
                streaming=True
            )
            self.logger.debug(f"Initialized OpenAI agent executor with agent: {self.agent}")
        elif self.agent == "Llama_index":
            self.llm = OpenAI(
                temperature=0,
                model=self.openai_model,
                api_key=self.api_key,
            )
            self.logger.debug(f"Initialized OpenAI agent executor with agent: {self.agent}")
        else:
            self.logger.error(f"Unsupported agent: {self.agent}")
            raise ValueError(f"Unsupported agent: {self.agent}")
        
        if agent == "Langchain":
            self.agent_executor = create_pandas_dataframe_agent(
                    self.llm,
                    self.df,
                    agent_type="tool-calling",
                    verbose=True,
                    allow_dangerous_code=True,
                    prefix=self.instruction,
                )
        elif agent == "Llama_index":
           self.agent_executor = PandasQueryEngine(
                    df=self.df,
                    verbose=True,
                    llm=self.llm
                )
        else:
            self.logger.error(f"Unsupported agent: {agent}")
            raise ValueError(f"Unsupported agent: {agent}")

    async def chat_with_a_df(self, question: str) -> str:
        self.logger.info(f"Received question: {question}")
        loop = asyncio.get_event_loop()
        try:
            if self.agent == "Llama_index":
                result = await self.agent_executor.ainvoke({"input": question})
                ans = result.get("output", "I don't know")
            else:
                ans = await loop.run_in_executor(None, self.agent_executor.query, question)
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
        # Ensure API key is set
        openai_key = os.getenv("OPENAI_API_KEY", "")

        # Initialize ChatwithCSV with OpenAI
        if not openai_key:
            logger.error("OPENAI_API_KEY is not set in environment variables.")
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
        chat = ChatwithCSV(api_key=openai_key, df=df)

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
