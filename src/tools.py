from langchain_community.tools import TavilySearchResults
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools import Tool

# Herramienta de búsqueda en internet
def get_internet_search_tool() -> Tool:
    
    internet_search = TavilySearchResults(max_results=10)
    internet_search.name = "internet_search"
    internet_search.description = "searchs the internet for recent information"
    return internet_search

# Herramienta de ejecución de código Python
def get_python_interpreter_tool() -> Tool:
    
    python_repl = PythonREPLTool()
    python_repl.name = "python_interpreter"
    python_repl.description = "executes Python code and returns the result"
    return python_repl

# Lista de herramientas que usará el agente
def get_tools() -> list[Tool]:
    return [
        get_internet_search_tool(),
        get_python_interpreter_tool()
    ]