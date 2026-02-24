# Classification and visualization prompts

CLASSIFICATION_PROMPT = """You are a message classifier. Analyze the user's message and determine:
1. Whether it is a casual chit-chat message (greetings, small talk) or a data query about the dataset
2. Whether the query would benefit from a visualization (typically when the answer involves tabular data, comparisons, distributions, or when the user explicitly asks for charts/graphs)

Respond in JSON format with the following structure:
{{
    "message_type": "chit_chat" or "data_query",
    "needs_visualization": true or false,
    "reasoning": "brief explanation of your classification"
}}

Examples:
- "Hello" -> {{"message_type": "chit_chat", "needs_visualization": false, "reasoning": "Simple greeting"}}
- "What percentage of passengers were male?" -> {{"message_type": "data_query", "needs_visualization": false, "reasoning": "Simple percentage question, no tabular output"}}
- "Show me a histogram of passenger ages" -> {{"message_type": "data_query", "needs_visualization": true, "reasoning": "Explicit request for visualization"}}
- "How many passengers embarked from each port?" -> {{"message_type": "data_query", "needs_visualization": true, "reasoning": "Tabular data output expected"}}

User message: {user_message}
"""

PLOTLY_GENERATION_PROMPT = """You are a data visualization expert. Generate Plotly code to create a visualization based on the user's query and the data output.

User Query: {user_query}

Data Output (from pandas query):
{data_output}

DataFrame Info:
{dataframe_info}

Instructions:
1. Generate valid Python code that uses plotly.graph_objects or plotly.express
2. The code should create a figure object (e.g., fig = px.bar(...) or fig = go.Figure(...))
3. Choose the most appropriate chart type:
   - Histogram for distributions
   - Bar chart for categorical comparisons
   - Pie chart for proportions
   - Line chart for trends
   - Scatter plot for relationships
4. Make the chart visually appealing with proper labels, titles, and colors
5. DO NOT include fig.show() - the figure will be displayed automatically
6. Use the 'df' variable which contains the full DataFrame - do not create sample data
7. Return ONLY the Python code, no explanations or markdown

Example output format:
```python
import plotly.express as px
fig = px.histogram(df, x='Age', title='Histogram of Ages')
```

Generate the Plotly code:
"""

CHIT_CHAT_RESPONSES = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello! What can I do for you?",
    "hey": "Hey there! How can I help?",
    "good morning": "Good morning! What can I assist you with?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How may I assist you?",
    "thanks": "You're welcome! Feel free to ask if you need anything else.",
    "thank you": "You're welcome! Happy to help!",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Take care!"
}
