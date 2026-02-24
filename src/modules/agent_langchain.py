# src/ChatwithCSV.py

import asyncio
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from .logging_config import get_logger  # Ensure correct relative import

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
            # Use ainvoke for async execution
            result = await self.agent_executor.ainvoke({"input": question})
            
            # Extract the output
            ans = result.get("output", "I don't know")
            
            # Extract the query/code executed and its output from intermediate steps
            query_executed = None
            query_output = None
            
            if "intermediate_steps" in result:
                intermediate_steps = result.get("intermediate_steps", [])
                self.logger.debug(f"Number of intermediate steps: {len(intermediate_steps)}")
                
                # Collect all queries executed (there might be multiple steps)
                queries = []
                outputs = []
                
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) >= 2:
                        # step is a tuple: (AgentAction, observation)
                        agent_action = step[0]
                        observation = step[1]
                        
                        # Extract the tool input (pandas code)
                        tool_input = None
                        if hasattr(agent_action, 'tool_input'):
                            tool_input = agent_action.tool_input
                        elif hasattr(agent_action, 'tool'):
                            # Sometimes the code is in the tool name or action
                            if hasattr(agent_action, 'action'):
                                tool_input = agent_action.action
                            else:
                                tool_input = str(agent_action.tool) if agent_action.tool else None
                        elif isinstance(agent_action, dict):
                            tool_input = agent_action.get('tool_input') or agent_action.get('action')
                        
                        if tool_input:
                            # If tool_input is a dict, extract the query
                            if isinstance(tool_input, dict):
                                query = tool_input.get('query') or tool_input.get('code') or str(tool_input)
                            else:
                                query = str(tool_input)
                            queries.append(query)
                        
                        # Extract the observation (query output)
                        if observation:
                            outputs.append(str(observation))
                
                # Use the last query and output (most relevant)
                if queries:
                    query_executed = queries[-1] if len(queries) == 1 else "\n".join([f"Step {i+1}:\n{q}" for i, q in enumerate(queries)])
                if outputs:
                    query_output = outputs[-1] if len(outputs) == 1 else "\n".join([f"Step {i+1}:\n{o}" for i, o in enumerate(outputs)])
                
                self.logger.debug(f"Query executed: {query_executed}")
                self.logger.debug(f"Query output: {query_output}")
            
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
