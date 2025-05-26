from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.tools import get_tools

def create_agent():
    # Inicializar el modelo LLM
    llm = ChatCohere()

    # Obtener las herramientas
    tools = get_tools()

    # Prompt mejorado para que el agente decida cuándo usar información en tiempo real
    system_prompt = '''
Eres un asistente útil.
- Si la pregunta requiere información actual, datos recientes, eventos, noticias, precios, clima, resultados deportivos, etc... 
mira primero en que año estamos usando la herramienta 'internet_search',
y si la fecha es superior a tus datos de entrenamiento ( diciembre de 2023) y la información
actual es distinta de la que tienes, DEBES usar la herramienta 'internet_search'.
- Si la pregunta es sobre conocimiento general, matemáticas, definiciones, o código, responde directamente o usa la herramienta 'python_interpreter' si es necesario.

Ejemplos:
Usuario: ¿Quién ganó el último mundial de fútbol?
Acción: internet_search

Usuario: ¿Cuál es la capital de Francia?
Respuesta directa: París

Usuario: ¿Qué tiempo hace hoy en Madrid?
Acción: internet_search

Usuario: ¿Cuánto es 2+2?
Respuesta directa: 4

Usuario: Hazme un gráfico de una función cuadrática.
Acción: python_interpreter

Usuario: ¿Cuál es el precio actual del Bitcoin?
Acción: internet_search

Usuario: ¿Quien es el penúltimo presidente de EEUU?
Acción: internet_search
'''
    prompt = ChatPromptTemplate.from_messages([
           ("system", f"{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
    ])

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
    )

    return agent_executor
