# LLM Agent Project
A Python project to build an agent using Cohere API and LangChain with a simple chatbox for the front-end with Gradio.

## Setup
1. Clone the repository.
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`.
3. Set up your `.env` file with `COHERE_API_KEY` and `TAVILY_API_KEY`.
4. Run the project: `python main.py`.

## Usage
The agent processes the query:
"Ask anything to the Agent"

It will:
- Search for the data online.

## Project Structure
- `src/tools.py`: Defines the tools (internet search and Python REPL) for the agent.
- `src/agent.py`: Creates the ReAct agent using Cohere and LangChain.
- `main.py`: Entry point to run the agent, it will deploy a local server with a chatbox.
- `requirements.txt`: Lists all dependencies.
- `.env`: Stores API keys (not included in the repository).