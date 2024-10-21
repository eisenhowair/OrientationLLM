from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput, Select
from prompt_warehouse import *
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from operator import itemgetter

import chainlit as cl
from chainlit.types import ThreadDict

from langchain.memory import ConversationBufferMemory


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("elias", "elias"):
        return cl.User(
            identifier="Elias", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


model = Ollama(base_url="http://localhost:11434", model="llama3:instruct")


@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            TextInput(id="formation_lvl", label="Post-"),
            Select(
                id="domaine",
                label="Domaine d'études de recherche",
                values=[
                    "",
                    "Informatique",
                    "Sports",
                    "Littérature",
                    "Chimie",
                ],
                initial_index=0,
            ),
        ]
    ).send()
    formation_lvl = settings["formation_lvl"]
    print(formation_lvl)
    cl.user_session.set(
        "memory", ConversationBufferMemory(return_messages=True))


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

    if settings["formation_lvl"] is not None:
        cl.user_session.set("formation", settings["formation_lvl"])
    if settings["domaine"] is not None:
        cl.user_session.set("domaine", settings["domaine"])


def setup_model(domaine, formation):
    """
    Configure le prompt et le Runnable pour conseiller l'utilisateur en fonction du domaine et de la formation de l'utilisateur.

    Args:
        domaine (str): Une chaîne représentant le domaine d'étude de l'utilisateur.
        formation (str): Une chaîne représentant le niveau de l'utilisateur.

    Returns:
        None : Met à jour la session utilisateur avec le nouveau Runnable pour discuter.
    """
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    # Initialiser le message spécifique
    specific_message = ""

    if domaine and formation:
        specific_message = f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}. L'utilisateur sort de la formation {formation}."
    elif domaine:
        specific_message = f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}."
    elif formation:
        specific_message = f"L'utilisateur sort de la formation {formation}."
    else:
        specific_message = prompt_no_domain_no_formation

    # Construire le prompt
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            ("system", f"{specific_message}"),
            MessagesPlaceholder(variable_name="history")
            #("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(
                memory.load_memory_variables) | itemgetter("history")
        ) 
        | prompt_exercice
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)



@cl.on_message
async def on_message(message: cl.Message):
    """
    Callback fonction appelée à chaque réception d'un message de l'utilisateur.
    Gère la logique principale de la conversation en fonction de l'état de la session utilisateur.

    Args:
        message (cl.Message): Le message envoyé par l'utilisateur.

    Returns:
        None : Envoie une réponse appropriée à l'utilisateur en fonction du contexte de la conversation.
    """
    formation = cl.user_session.get("formation")
    domaine = cl.user_session.get("domaine")
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    setup_model(domaine=domaine,formation=formation)
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()

    # on ajoute les messages à la mémoire
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)

    print(memory.load_memory_variables)
  

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    await cl.ChatSettings(
        [
            TextInput(id="formation_lvl", label="Post-"),
            Select(
                id="domaine",
                label="Domaine d'études de recherche",
                values=[
                    "",
                    "Informatique",
                    "Sports",
                    "Littérature",
                    "Chimie",
                ],
                initial_index=0,
            ),
        ]
    ).send()


    formation = cl.user_session.get("formation")
    domaine = cl.user_session.get("domaine")

    setup_model(domaine=domaine,formation=formation)