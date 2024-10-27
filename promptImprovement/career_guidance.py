from langchain_community.llms import Ollama
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput, Select, Tags
from prompt_warehouse import *
from prepare_prompt import *

import chainlit as cl
from chainlit.types import ThreadDict

from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("Elias", "pass"):
        return cl.User(
            identifier="Elias", metadata={"role": "admin", "provider": "credentials"}
        )
    elif (username, password) == ("Théo", "pass"):
        return cl.User(
            identifier="Théo", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


# model = Ollama(base_url="http://localhost:11434", model="llama3.1:8b-instruct-q4_1")


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
            Select(
                id="model_choice",
                label="Choix du modèle",
                values=[
                    "",
                    "llama3.1:8b-instruct-q4_1",
                    "llama3:instruct",
                    "llama3.2:3b-instruct-q8_0",
                    "nemotron-mini",
                    "nemotron-mini:4b-instruct-q5_0",
                ],
                initial_index=0,
            ),
        ]
    ).send()

    old_settings = {"formation_lvl": None, "domaine": None, "model_choice": None}
    print(old_settings)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("old_settings", old_settings)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    old_settings = cl.user_session.get("old_settings")
    print("voila les vieux settings:", old_settings)

    # Vérifier les changements et mettre à jour si nécessaire
    changes_made = False

    if settings["formation_lvl"] != old_settings["formation_lvl"]:
        changes_made = True
        cl.user_session.set("formation", settings["formation_lvl"])

    if settings["domaine"] != old_settings["domaine"]:
        changes_made = True
        cl.user_session.set("domaine", settings["domaine"])

    if settings["model_choice"] != old_settings["model_choice"]:
        changes_made = True
        cl.user_session.set("model", settings["model_choice"])

    if changes_made:
        # Mettre à jour old_settings avec une nouvelle copie des settings actuels
        new_old_settings = {
            "formation_lvl": settings["formation_lvl"],
            "domaine": settings["domaine"],
            "model_choice": settings["model_choice"],
        }
        cl.user_session.set("old_settings", new_old_settings)


def setup_model(domaine, formation, nom_model):
    """
    Configure le prompt et le Runnable pour conseiller l'utilisateur en fonction du domaine et de la formation de l'utilisateur.

    Args:
        domaine (str): Une chaîne représentant le domaine d'étude de l'utilisateur.
        formation (str): Une chaîne représentant le niveau de l'utilisateur.

    Returns:
        None : Met à jour la session utilisateur avec le nouveau Runnable pour discuter.
    """

    # Initialiser le message spécifique
    specific_message = ""

    if domaine and formation:
        specific_message = f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}. L'utilisateur sort de la formation {formation}."
    elif domaine:
        specific_message = f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}."
    elif formation:
        specific_message = f"L'utilisateur sort de la formation {formation}."
    else:
        specific_message = prompt_no_domain_no_formation_v3

    model = Ollama(
        base_url="http://localhost:11434",
        model=nom_model,
        cache=True,
        num_ctx=32768,
        repeat_penalty=1.3,
    )
    runnable = prepare_prompt_zero_shot(corps_prompt=specific_message, model=model)
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
    model = cl.user_session.get("model")
    print("voilà le modèle actuel:", model)
    if model is None:
        await cl.Message(
            content="Merci de sélectionner un modèle dans les options ci-dessous."
        ).send()
    else:
        setup_model(domaine=domaine, formation=formation, nom_model=model)
        runnable = cl.user_session.get("runnable")  # type: Runnable

        msg = cl.Message(content="")
        async for chunk in runnable.astream(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        await msg.send()

        # on ajoute les messages à la mémoire
        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(msg.content)

    print(memory.load_memory_variables)  # affiche la discussion


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
            Select(
                id="model_choice",
                label="Choix du modèle",
                values=[
                    "",
                    "llama3.1:8b-instruct-q4_1",
                    "llama3:instruct",
                    "llama3.2:3b-instruct-q8_0",
                    "nemotron-mini",
                    "nemotron-mini:4b-instruct-q5_0",
                ],
                initial_index=0,
            ),
        ]
    ).send()

    old_settings = {"formation_lvl": None, "domaine": None, "model_choice": None}
    print("old_settings on chat resume:", old_settings)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("old_settings", old_settings)

    formation = cl.user_session.get("formation")
    domaine = cl.user_session.get("domaine")

    # setup_model(domaine=domaine, formation=formation)
