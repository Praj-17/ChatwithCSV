# ChatwithCSV

**Interactive CSV Q&A Chatbot** — Ask questions in natural language about your CSV data and get answers, with optional charts and visualizations. Powered by OpenAI and LangChain.

---

## What This Project Does

ChatwithCSV is a Streamlit app that lets you:

- **Chat with your data** — Upload a CSV (or use the built-in Titanic dataset) and ask questions in plain English.
- **Get instant answers** — A LangChain agent runs pandas queries behind the scenes and returns clear, conversational answers.
- **See the code** — Expand "View Query Executed" to see the exact Python/pandas code that was run.
- **View visualizations** — Ask for charts (histograms, bar charts, pie charts, etc.) and get interactive Plotly charts in the chat.
- **Handle chit-chat** — Say "Hello", "Thanks", or "Bye" and get friendly replies without hitting the data pipeline.

The app uses a **classifier** to decide whether your message is small talk or a data question, and whether to generate a chart. Data questions are handled by a **pandas DataFrame agent**; when a chart is needed, a **Plotly tool** generates and renders the figure.

---

## Screenshots

*(Add your own screenshots to the `static/` folder and reference them here. Suggested filenames below.)*

| Description | Screenshot |
|-------------|------------|
| **Chat interface** — Sidebar with upload, dataset preview, and chat tab | ![Chat interface](static/chat-interface.png) |
| **Sample Q&A** — Natural language question and answer with expandable query/code | ![Q&A example](static/qa-example.png) |
| **Visualization** — Plotly chart (e.g. histogram or bar chart) in the chat | ![Visualization](static/visualization.png) |
| **Sample Queries tab** — List of example questions and answers | ![Sample queries](static/sample-queries.png) |
| **FAQs and Contact** — FAQs tab and contact information | ![FAQs](static/faqs.png) |

*If you don't have screenshots yet, add PNG/JPG files to `static/` with the names above (e.g. `chat-interface.png`) and they will show up here.*

---

## Features

- **Default dataset** — Titanic CSV is loaded by default so you can try the app without uploading a file.
- **Optional custom CSV** — Upload your own CSV in the sidebar to replace the default dataset.
- **LangChain + OpenAI** — Uses `gpt-4o-mini` for the agent and classifier (via LiteLLM).
- **Pandas DataFrame agent** — Executes pandas code safely and returns results.
- **Plotly visualizations** — Histograms, bar charts, pie charts, and more, generated from your data.
- **Chat context** — Recent conversation history is passed to the agent for follow-up questions.
- **Tabs** — Chat, FAQs, Sample Queries, and Contact in a single app.

---

## Sample Questions

You can try these with the default Titanic dataset (or your own CSV with similar columns):

| Category | Sample question |
|----------|-----------------|
| **Stats** | What percentage of passengers were male on the Titanic? |
| **Stats** | What was the average ticket fare? |
| **Visualization** | Show me a histogram of passenger ages |
| **Visualization** | Show me a histogram of passenger ages by gender |
| **Visualization** | How many passengers embarked from each port? |
| **Visualization** | Create a bar chart of passengers by embarkation port |
| **Visualization** | What was the survival rate by passenger class? |
| **Visualization** | Show me a pie chart of survival rates |
| **Visualization** | How many passengers survived versus did not survive? |
| **Visualization** | Create a visualization showing ticket fare distribution |
| **Visualization** | Show me a comparison chart of survival rates by gender and class |

*More examples and pre-written answers are available in the **Sample Queries** tab inside the app.*

---

## What's Included in This Project

- **Frontend** — Streamlit UI (chat, sidebar, tabs).
- **Agent** — LangChain pandas DataFrame agent with OpenAI.
- **Classifier** — Detects chit-chat vs data query and whether a chart is needed.
- **Plotly tool** — Generates Plotly code from the LLM and renders figures in the app.
- **Prompts** — Centralized in `src/constants/prompts.py` (classification, Plotly generation, chit-chat replies).
- **Sample data** — Default Titanic CSV at `src/data/titanic.csv`.
- **Sample queries** — JSON at `src/constants/sample_queries.json` used in the Sample Queries tab.
- **Docker** — Dockerfile for running the app in a container.
- **Logging** — Module-level logging for debugging.

---

## Setup Instructions

### Prerequisites

- **Python 3.10+**
- **OpenAI API key** — Get one from [OpenAI API Keys](https://platform.openai.com/settings/organization/api-keys).

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ChatwithCSV
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The app loads this via `python-dotenv`. If the key is missing, Streamlit will show an error and stop.

### 5. Run the app

```bash
streamlit run main.py
```

Or:

```bash
python -m streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Running with Docker

### Build the image

```bash
docker build -t chatwithcsv .
```

### Run the container

Pass your API key as an environment variable:

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_openai_api_key_here chatwithcsv
```

Or use a `.env` file (ensure it's not in `.dockerignore` if you copy it):

```bash
docker run -p 8501:8501 --env-file .env chatwithcsv
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure

```
ChatwithCSV/
├── main.py                 # Streamlit app entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
├── .env                    # OPENAI_API_KEY (create this yourself)
├── static/                 # Screenshots for README (e.g. chat-interface.png)
├── src/
│   ├── __init__.py         # Exposes ChatwithCSV, get_logger
│   ├── constants/
│   │   ├── prompts.py      # Classification, Plotly, chit-chat prompts
│   │   └── sample_queries.json
│   ├── data/
│   │   └── titanic.csv     # Default dataset
│   └── modules/
│       ├── agent_langchain.py   # ChatwithCSV, pandas agent, Plotly tool wiring
│       ├── classifier_agent.py  # Chit-chat vs data query, needs_visualization
│       ├── plotly_tool.py       # Plotly code generation and execution
│       └── logging_config.py    # Logger setup
```

---

## FAQs

| Question | Answer |
|----------|--------|
| **What CSV is used by default?** | The app uses `src/data/titanic.csv`. You can upload your own CSV in the sidebar to replace it. |
| **How do I upload my own CSV?** | Use the "Upload Custom CSV File (Optional)" in the sidebar. It will replace the default dataset. |
| **How do I ask questions?** | Type in the chat input at the bottom of the **Chat** tab. |
| **How do I set the OpenAI API key?** | Create a `.env` file in the project root with `OPENAI_API_KEY=your_key`. |
| **How do I create an OpenAI API key?** | Go to [OpenAI API Keys](https://platform.openai.com/settings/organization/api-keys), sign in, and create a new secret key. |
| **What model is used?** | The app uses **GPT-4o-mini** for the LangChain agent, classifier, and Plotly code generation (via LiteLLM). |

---

## Contact

For questions or feedback:

- **Email:** pwaykos1@gmail.com  
- **LinkedIn:** [Prajwal Waykos](https://www.linkedin.com/in/prajwal-waykos/)  
- **GitHub:** [praj-17](https://github.com/praj-17)  

---

## License

Use and modify as needed for your own projects.
