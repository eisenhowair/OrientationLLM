from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput, Select

import chainlit as cl

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
    # Message système de base
    system_message = "Tu parles uniquement français."

    # Initialiser le message spécifique
    specific_message = ""

    if domaine and formation:
        specific_message = f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}. L'utilisateur sort de la formation {formation}."
    elif domaine:
        specific_message = f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}."
    elif formation:
        specific_message = f"L'utilisateur sort de la formation {formation}."
    else:
        specific_message = "Ton rôle est de conseiller l'utilisateur sans information spécifique."

    # Construire le prompt
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            ("system", f"{system_message} {specific_message}"),
            ("human", "{question}")
        ]
    )

    runnable = prompt_exercice | model | StrOutputParser()
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

    setup_model(domaine=domaine,formation=formation)
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()
    print(message.content)
  
