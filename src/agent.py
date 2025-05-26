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
            ("system", "Answer the following questions as best you can, with the available tools."),
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
