# src/ChatwithCSV.py

import asyncio
import json
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from .logging_config import get_logger  # Ensure correct relative import
from .plotly_tool import PlotlyVisualizationTool


def _extract_json_from_observation(observation_str: str):
    """Extract the first complete JSON object from a string (handles trailing text from agent)."""
    s = str(observation_str).strip()
    start = s.find("{")
    if start == -1:
        return None
    # Count braces but ignore those inside double-quoted strings
    depth = 0
    i = start
    in_string = False
    escape = False
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
        i += 1
    return None


def _get_observation_string(observation):
    """Get the raw string from observation (handles ToolMessage and plain string)."""
    if hasattr(observation, "content") and isinstance(observation.content, str):
        return observation.content.strip()
    return str(observation).strip()


def _format_chat_history_for_input(chat_history: list, current_question: str, max_history_chars: int = 2500) -> str:
    """Format chat history and current question into a single input string for the agent."""
    if not chat_history:
        return current_question
    lines = ["Previous conversation:"]
    total = 0
    for m in chat_history:
        if total >= max_history_chars:
            break
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        line = f"{prefix}: {content}"
        if total + len(line) > max_history_chars:
            line = line[: max_history_chars - total - 4] + "..."
        lines.append(line)
        total += len(line)
    lines.append("")
    lines.append(f"Current question: {current_question}")
    return "\n".join(lines)


class QueryCaptureCallback(BaseCallbackHandler):
    """Callback to capture the pandas query executed and its output (not the plotly tool)."""
    def __init__(self):
        self.query_executed = None
        self.query_output = None
        self.logger = None
        self._skip_next_output = False
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Capture the tool input only for the pandas/repl tool, not the plotly tool."""
        tool_name = (serialized.get("name") or "").lower()
        self._skip_next_output = "plotly" in tool_name
        if self._skip_next_output:
            return
        if isinstance(input_str, dict):
            self.query_executed = input_str.get('query') or str(input_str)
        else:
            self.query_executed = str(input_str)
        if self.logger:
            self.logger.debug(f"Captured query: {self.query_executed}")
    
    def on_tool_end(self, output, **kwargs):
        """Capture the tool output only for the pandas/repl tool."""
        if getattr(self, '_skip_next_output', False):
            self._skip_next_output = False
            return
        self._skip_next_output = False
        self.query_output = str(output)
        if self.logger:
            self.logger.debug(f"Captured output (first 200 chars): {self.query_output[:200]}")

class ChatwithCSV:
    def __init__(self, api_key: str, df: pd.DataFrame, needs_visualization: bool = False) -> None:
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.df = df
        self.openai_model = "gpt-4o-mini"
        self.needs_visualization = needs_visualization
        
        # Initialize plotly tool if visualization is needed
        self.plotly_tool = None
        self.plotly_tool_instance = None
        if needs_visualization:
            self.plotly_tool_instance = PlotlyVisualizationTool(api_key=api_key)
            self.plotly_tool = self.plotly_tool_instance.create_langchain_tool(df=df)
        
        # Update instruction based on visualization capability
        if needs_visualization:
            self.instruction = (
                "You are an excellent data analyst who can answer questions based on a given pandas dataframe. "
                "You have two kinds of tools: (1) a Python REPL to run pandas queries on the dataframe 'df', and (2) generate_plotly_visualization to create charts. "
                "When the user asks for a visualization, chart, graph, or histogram you MUST do the following in order: "
                "STEP 1: Run a pandas query to get the relevant data (e.g. for a histogram of ages run df['Age'].dropna(); for counts by category run the appropriate groupby/value_counts). "
                "Use the full dataset so the chart is meaningful, not just a few rows. "
                "STEP 2: Call generate_plotly_visualization with 'query' (user question), 'data_output' (the exact string output from your pandas query in STEP 1), and 'dataframe_info' (e.g. df.columns.tolist() and df.info() or describe). "
                "Do not call generate_plotly_visualization with only a sample or head(); always pass the result of a query that gets the data needed for the chart. "
                "After the visualization tool succeeds, respond with ONE short sentence (e.g. 'Here is the histogram of ages.'). "
                "Do not generate or embed any image or base64 in your response; the app will display the chart. "
                "If you cannot figure out the answer, say `The given context does not provide answer to the following problem`."
            )
        else:
            self.instruction = (
                "You are an excellent data analyst who can answer questions based on a given pandas dataframe. "
                "If you cannot figure out the answer, just politely say `The given context does not provide answer to the following problem`."
            )
        
        self.logger.debug(f"Initializing ChatwithCSV with OpenAI and Langchain agent (visualization: {needs_visualization})")

        self.llm = ChatOpenAI(
            temperature=0,
            model=self.openai_model,
            openai_api_key=self.api_key,
            streaming=True
        )
        
        # Create agent with optional plotly tool (max_iterations to avoid timeout)
        agent_kwargs = {
            "llm": self.llm,
            "df": self.df,
            "agent_type": "tool-calling",
            "verbose": True,
            "allow_dangerous_code": True,
            "prefix": self.instruction,
            "max_iterations": 10,
            "max_execution_time": 90.0,
        }
        
        # Add extra_tools if visualization is needed
        if self.plotly_tool:
            agent_kwargs["extra_tools"] = [self.plotly_tool]
            self.logger.debug("Added Plotly visualization tool to agent")
        
        self.agent_executor = create_pandas_dataframe_agent(**agent_kwargs)
        self.logger.debug("Initialized OpenAI agent executor with Langchain")

    async def chat_with_a_df(self, question: str, chat_history: list = None) -> dict:
        """
        Process a question and return both the answer and execution details.
        
        Args:
            question: The user's current question.
            chat_history: Optional list of previous messages [{"role": "user"|"assistant", "content": "..."}] for context.
        
        Returns:
            dict: Contains 'answer', 'query_executed', 'query_output', 'visualization_figure', 'needs_visualization'
        """
        chat_history = chat_history or []
        agent_input = _format_chat_history_for_input(chat_history, question)
        self.logger.info(f"Received question: {question}")
        try:
            # Create callback to capture query and output
            callback = QueryCaptureCallback()
            callback.logger = self.logger
            
            # Use ainvoke with callback to capture intermediate steps
            # Add timeout to prevent hanging
            try:
                result = await asyncio.wait_for(
                    self.agent_executor.ainvoke(
                        {"input": agent_input},
                        config={"callbacks": [callback]}
                    ),
                    timeout=120.0  # 2 minute timeout
                )
            except asyncio.TimeoutError:
                self.logger.error("Agent execution timed out after 120 seconds")
                return {
                    "answer": "I'm sorry, the request took too long to process. Please try a simpler query.",
                    "query_executed": None,
                    "query_output": None,
                    "visualization_figure": None,
                    "plotly_code": None,
                    "needs_visualization": False
                }
            
            # Extract the output
            ans = result.get("output", "I don't know")
            
            # Get captured query and output from callback
            query_executed = callback.query_executed
            query_output = callback.query_output
            
            # Check for plotly visualization in intermediate steps
            visualization_figure = None
            plotly_code = None
            
            if "intermediate_steps" in result:
                intermediate_steps = result.get("intermediate_steps", [])
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) >= 2:
                        agent_action = step[0]
                        observation = step[1]
                        
                        # Check if this step used the plotly tool
                        if hasattr(agent_action, 'tool') and 'plotly' in str(agent_action.tool).lower():
                            try:
                                observation_str = _get_observation_string(observation)
                                self.logger.debug(f"Plotly tool observation: {observation_str[:200]}")
                                # Try direct parse first, then extract in case of trailing text
                                plotly_result = None
                                try:
                                    plotly_result = json.loads(observation_str)
                                except json.JSONDecodeError:
                                    plotly_result = _extract_json_from_observation(observation_str)
                                if not plotly_result:
                                    raise ValueError("Could not extract JSON from observation")
                                if plotly_result.get("success") and plotly_result.get("plotly_code"):
                                    plotly_code = plotly_result.get("plotly_code")
                                    
                                    # Regenerate figure from code (no caching)
                                    self.logger.info("Regenerating figure from code")
                                    try:
                                        if self.plotly_tool_instance:
                                            fig = self.plotly_tool_instance.execute_plotly_code(plotly_code, self.df)
                                        else:
                                            plotly_tool_instance = PlotlyVisualizationTool(api_key=self.api_key)
                                            fig = plotly_tool_instance.execute_plotly_code(plotly_code, self.df)
                                        
                                        if fig is not None:
                                            visualization_figure = fig
                                            self.logger.info("Successfully generated plotly visualization")
                                        else:
                                            self.logger.warning("Plotly code executed but no figure returned")
                                    except Exception as e:
                                        self.logger.error(f"Error regenerating figure: {e}")
                                        import traceback
                                        self.logger.error(traceback.format_exc())
                                        # Do not overwrite visualization_figure with None if we already have one
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Could not parse plotly result as JSON: {e}")
                                self.logger.debug(f"Observation was: {str(observation)[:500]}")
                            except ValueError as e:
                                self.logger.warning(f"Could not extract JSON from observation: {e}")
                            except Exception as e:
                                self.logger.error(f"Error extracting plotly visualization: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
            
            # Fallback: Try to get query_executed from intermediate_steps if callback didn't capture
            if not query_executed and "intermediate_steps" in result:
                intermediate_steps = result.get("intermediate_steps", [])
                if intermediate_steps:
                    for step in intermediate_steps:
                        if isinstance(step, tuple) and len(step) >= 2:
                            agent_action = step[0]
                            if hasattr(agent_action, 'tool_input'):
                                tool_input = agent_action.tool_input
                                # Skip plotly tool inputs
                                if hasattr(agent_action, 'tool') and 'plotly' not in str(agent_action.tool).lower():
                                    if isinstance(tool_input, dict):
                                        query_executed = tool_input.get('query') or str(tool_input)
                                    else:
                                        query_executed = str(tool_input)
                                    query_output = str(step[1]) if step[1] else None
                                    break
            
            self.logger.info(f"Query executed: {query_executed}")
            self.logger.info(f"Query output (length): {len(query_output) if query_output else 0}")
            self.logger.info(f"Visualization generated: {visualization_figure is not None}")

            # Force visualization when classifier said it's needed but agent didn't produce a figure
            if (
                self.needs_visualization
                and visualization_figure is None
                and self.plotly_tool_instance is not None
                and query_output
                and question
            ):
                self.logger.info("Forcing visualization: generating chart from query result")
                try:
                    dataframe_info = f"Columns: {self.df.columns.tolist()}\nShape: {self.df.shape}\nDtypes:\n{self.df.dtypes}"
                    code = self.plotly_tool_instance.generate_plotly_code(
                        user_query=question,
                        data_output=query_output[:2000],
                        dataframe_info=dataframe_info,
                    )
                    if code:
                        fig = self.plotly_tool_instance.execute_plotly_code(code, self.df)
                        if fig is not None:
                            visualization_figure = fig
                            plotly_code = code
                            self.logger.info("Forced visualization generated successfully")
                        else:
                            self.logger.warning("Forced visualization: execute_plotly_code returned None")
                    else:
                        self.logger.warning("Forced visualization: generate_plotly_code returned None")
                except Exception as e:
                    self.logger.error(f"Error in forced visualization: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            self.logger.debug(f"Response: {ans}")
            
            return {
                "answer": ans,
                "query_executed": query_executed,
                "query_output": query_output,
                "visualization_figure": visualization_figure,
                "plotly_code": plotly_code,
                "needs_visualization": self.needs_visualization
            }
        except Exception as e:
            self.logger.error(f"Error in chat_with_a_df: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "answer": "I'm sorry, I couldn't process your request.",
                "query_executed": None,
                "query_output": None,
                "visualization_figure": None,
                "plotly_code": None,
                "needs_visualization": False
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
