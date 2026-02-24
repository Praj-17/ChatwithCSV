# src/ChatwithCSV.py

import asyncio
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from .logging_config import get_logger  # Ensure correct relative import

class QueryCaptureCallback(BaseCallbackHandler):
    """Callback to capture the query executed and its output."""
    def __init__(self):
        self.query_executed = None
        self.query_output = None
        self.logger = None
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Capture the tool input (query)."""
        if isinstance(input_str, dict):
            self.query_executed = input_str.get('query') or str(input_str)
        else:
            self.query_executed = str(input_str)
        if self.logger:
            self.logger.debug(f"Captured query: {self.query_executed}")
    
    def on_tool_end(self, output, **kwargs):
        """Capture the tool output."""
        self.query_output = str(output)
        if self.logger:
            self.logger.debug(f"Captured output (first 200 chars): {self.query_output[:200]}")

class ChatwithCSV:
    def __init__(self, api_key: str, df: pd.DataFrame) -> None:
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.df = df
        self.openai_model = "gpt-4o-mini"
        self.instruction = (
            "You are an excellent data analyst who can answer questions based on a given pandas dataframe. "
            "If you cannot figure out the answer, just politely say `The given context does not provide answer to the following problem`."
        )
        self.logger.debug("Initializing ChatwithCSV with OpenAI and Langchain agent")

        self.llm = ChatOpenAI(
            temperature=0,
            model=self.openai_model,
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
        self.logger.debug("Initialized OpenAI agent executor with Langchain")

    async def chat_with_a_df(self, question: str) -> dict:
        """
        Process a question and return both the answer and execution details.
        
        Returns:
            dict: Contains 'answer', 'query_executed', and 'query_output'
        """
        self.logger.info(f"Received question: {question}")
        try:
            # Create callback to capture query and output
            callback = QueryCaptureCallback()
            callback.logger = self.logger
            
            # Use ainvoke with callback to capture intermediate steps
            result = await self.agent_executor.ainvoke(
                {"input": question},
                config={"callbacks": [callback]}
            )
            
            # Extract the output
            ans = result.get("output", "I don't know")
            
            # Get captured query and output from callback
            query_executed = callback.query_executed
            query_output = callback.query_output
            
            # Fallback: Try to get from result if callback didn't capture
            if not query_executed and "intermediate_steps" in result:
                intermediate_steps = result.get("intermediate_steps", [])
                if intermediate_steps:
                    last_step = intermediate_steps[-1]
                    if isinstance(last_step, tuple) and len(last_step) >= 2:
                        agent_action = last_step[0]
                        if hasattr(agent_action, 'tool_input'):
                            tool_input = agent_action.tool_input
                            if isinstance(tool_input, dict):
                                query_executed = tool_input.get('query') or str(tool_input)
                            else:
                                query_executed = str(tool_input)
                        query_output = str(last_step[1]) if last_step[1] else None
            
            self.logger.info(f"Query executed: {query_executed}")
            self.logger.info(f"Query output (length): {len(query_output) if query_output else 0}")
            
            self.logger.debug(f"Response: {ans}")
            
            return {
                "answer": ans,
                "query_executed": query_executed,
                "query_output": query_output
            }
        except Exception as e:
            self.logger.error(f"Error in chat_with_a_df: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "answer": "I'm sorry, I couldn't process your request.",
                "query_executed": None,
                "query_output": None
            }


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
