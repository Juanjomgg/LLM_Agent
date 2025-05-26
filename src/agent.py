from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from src.tools import get_tools

def create_agent():
    # Inicializar el modelo LLM
    llm = ChatCohere()

    # Obtener las herramientas
    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant. 
             For questions requiring real-time information (like the current date, recent events, or up-to-date data),"
            "use the 'internet_search' tool to search the web. If the task involves calculations or generating code (like plotting data),"
            "use the 'python_interpreter' tool to execute Python code. Always provide accurate and up-to-date answers."""),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    # Crear el agente
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # Crear el ejecutor del agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False # Muestra el razonamiento del agente en la consola
    )

    return agent_executor
