# Plotly visualization tool for LangChain

import json
import litellm
import pandas as pd
from typing import Dict, Optional, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from .logging_config import get_logger
from ..constants.prompts import PLOTLY_GENERATION_PROMPT

class PlotlyVisualizationTool:
    """Tool for generating Plotly visualizations using LLM."""
    
    def __init__(self, api_key: str):
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.model = "gpt-4o-mini"
        
        # Set OpenAI API key for litellm
        litellm.api_key = api_key
    
    def generate_plotly_code(self, user_query: str, data_output: str, dataframe_info: str) -> str:
        """
        Generate Plotly code using LLM based on user query and data output.
        
        Args:
            user_query: The user's original query
            data_output: The output from the pandas query (can be string representation of DataFrame or Series)
            dataframe_info: Information about the dataframe structure
            
        Returns:
            str: Python code that generates a Plotly figure
        """
        self.logger.debug(f"Generating plotly code for query: {user_query[:100]}")
        
        try:
            # Format the prompt
            prompt = PLOTLY_GENERATION_PROMPT.format(
                user_query=user_query,
                data_output=str(data_output)[:2000],  # Limit length to avoid token issues
                dataframe_info=dataframe_info
            )
            
            # Call OpenAI via litellm
            response = litellm.completion(
                model=f"openai/{self.model}",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Generate only valid Python code using Plotly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Slight creativity for better visualizations
            )
            
            # Extract the response
            code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
            # Remove fig.show() if present (not needed for Streamlit)
            code = code.replace("fig.show()", "").strip()
            # Remove any trailing semicolons or empty lines
            code = code.rstrip(";").strip()
            
            self.logger.debug(f"Generated plotly code (first 200 chars): {code[:200]}")
            
            return code
            
        except Exception as e:
            self.logger.error(f"Error generating plotly code: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def execute_plotly_code(self, code: str, df: pd.DataFrame) -> Optional[Any]:
        """
        Execute the generated Plotly code and return the figure object.
        
        Args:
            code: The generated Plotly Python code
            df: The pandas DataFrame to use in the code
            
        Returns:
            Plotly figure object or None if execution fails
        """
        if not code:
            return None
            
        self.logger.debug("Executing plotly code")
        
        try:
            # Create a safe execution environment
            local_vars = {
                'df': df,
                'pd': pd,
                'px': None,
                'go': None,
                'fig': None
            }
            
            # Import plotly in the execution context
            import plotly.express as px
            import plotly.graph_objects as go
            local_vars['px'] = px
            local_vars['go'] = go
            
            # Execute the code
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Get the figure object
            fig = local_vars.get('fig')
            
            if fig is None:
                self.logger.warning("Plotly code executed but 'fig' variable not found")
                return None
            
            self.logger.info("Successfully generated plotly figure")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error executing plotly code: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_langchain_tool(self, df: pd.DataFrame) -> StructuredTool:
        """
        Create a LangChain StructuredTool for Plotly visualization.
        
        Args:
            df: The pandas DataFrame
            
        Returns:
            LangChain StructuredTool object
        """
        # Define the input schema for the tool
        class PlotlyToolInput(BaseModel):
            query: str = Field(description="The user's original question or query")
            data_output: str = Field(description="The output from the pandas query (string representation of DataFrame or Series)")
            dataframe_info: str = Field(description="Information about the dataframe structure (columns, rows, etc.)")
        
        def plotly_tool_func(query: str, data_output: str, dataframe_info: str) -> str:
            """
            Tool function that generates and executes plotly code.
            
            Args:
                query: The user's original question
                data_output: The output from the pandas query
                dataframe_info: Information about the dataframe structure
            """
            try:
                self.logger.debug(f"Plotly tool called with query: {query[:100]}")
                
                # Generate plotly code
                code = self.generate_plotly_code(query, data_output, dataframe_info)
                
                if not code:
                    return "Failed to generate plotly code"
                
                # Execute code to verify it works (but don't store the figure)
                fig = self.execute_plotly_code(code, df)
                
                if fig is None:
                    return "Failed to execute plotly code"
                
                # Don't cache the figure - we'll regenerate it from code when needed
                # Just verify the code works, then return the code
                self.logger.info("Plotly tool completed successfully, returning response")
                response_data = {
                    "plotly_code": code,
                    "success": True,
                    "message": "Plotly visualization generated successfully"
                }
                return json.dumps(response_data)
                
            except Exception as e:
                self.logger.error(f"Error in plotly_tool_func: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return f"Error: {str(e)}"
        
        return StructuredTool.from_function(
            func=plotly_tool_func,
            name="generate_plotly_visualization",
            description="""Generates interactive Plotly visualizations (charts, histograms, bar charts, pie charts). 
            Use ONLY after you have run a pandas query to get the data. Pass 'query' (user question), 'data_output' (the exact string result from your pandas query - e.g. df['Age'].dropna() or value_counts()), and 'dataframe_info' (e.g. df.columns.tolist() or df.info()). 
            Do not pass a sample or head(); pass the full result of the query that gets the data for the chart. The app renders the figure automatically.""",
            args_schema=PlotlyToolInput
        )
