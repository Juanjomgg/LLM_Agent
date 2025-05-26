import os
from dotenv import load_dotenv
from src.agent import create_agent
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

# Variables de entorno
load_dotenv()

# Instacia del ejecutador del agente
agent_executor = create_agent()

# --- Función para el manejo de la conversación con Gradio ---
# Gradio pasará el mensaje actual y el historial completo de la conversación
def respond_to_chat(message: str, history: list[list[str]]):

    # Necesitamos convertirlo a la lista de HumanMessage/AIMessage que espera el agente de LangChain.
    langchain_chat_history = []
    for human_msg, ai_msg in history:
        langchain_chat_history.append(HumanMessage(content=human_msg))
        if ai_msg is not None: # Asegurarse de que el mensaje del AI no sea None
            langchain_chat_history.append(AIMessage(content=ai_msg))

    try:
        # Invocar al agente con el mensaje actual del usuario y el historial formateado
        response = agent_executor.invoke(
            {
                "input": message,
                "chat_history": langchain_chat_history
            }
        )
        agent_response = response["output"]
        return agent_response
    except Exception as e:
        print(f"Error durante la invocación del agente: {e}")
        return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"

# --- Configuración e inicio de la Interfaz de Gradio ---
def main():
    # Creamos la interfaz de chat de Gradio
    iface = gr.ChatInterface(
        fn=respond_to_chat, # La función que Gradio llamará para cada mensaje
        title="🤖 Asistente LangChain",
        chatbot=gr.Chatbot(height=500, label="Conversación"), # Área de visualización de la conversación
    )

    # Lanza la interfaz de Gradio
    iface.launch()

if __name__ == "__main__":
    main()

