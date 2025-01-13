from langchain_ollama.llms import OllamaLLM
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput, Select
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))

from ..prompt_warehouse import (
    prompt_v3_context,
    prompt_v3_no_context,
    prompt_v4_context,
    prompt_v4_context_strict,
)

import chainlit as cl
from chainlit.types import ThreadDict
from langchain.memory import ConversationBufferMemory
from ensemble_model_gestion import EnsembleModelManager
from typing import List, Dict, Optional
from dotenv import load_dotenv
from vector_store_manager import VectorStoreFAISS

# Pour ajouter un modèle: le rajouter dans les available model de ensemble_model_gestion.py


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


@cl.on_chat_start
async def on_chat_start():

    # mise en place des settings
    await cl.ChatSettings(
        [
            TextInput(id="formation_lvl", label="Post-"),
            TextInput(id="domaine", label="Domaine d'études de recherche"),
        ]
    ).send()

    old_settings = {
        "formation_lvl": None,
        "domaine": None,
    }

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("old_settings", old_settings)
    cl.user_session.set(
        "corps_prompt",
        generate_specific_message(None, None),
    )

    load_dotenv()
    ensemble_model_path = os.getenv("VECTORSTORE_INDEX_PATH")
    if not ensemble_model_path:
        raise ValueError(
            "La variable d'environnement VECTORSTORE_INDEX_PATH n'est pas définie."
        )

    # mise en place du RAG
    vectorstore = VectorStoreFAISS(
        embedding_model_name="hkunlp/instructor-large",
        index_path=ensemble_model_path,  # "embedding_indexes"
    )

    cl.user_session.set("vectorstore", vectorstore)

    # Initialize EnsembleModelManager
    ensemble_manager = EnsembleModelManager()
    cl.user_session.set("ensemble_manager", ensemble_manager)
    ensemble_manager.activate_models(list(ensemble_manager.available_models.keys()))
    runnable = ensemble_manager.create_ensemble_runnable(
        domaine=None,
        formation=None,
        use_few_shot=False,
    )

    cl.user_session.set("runnable", runnable)

    # mise en place du RAG

    """
    rag_chain = (
        {"context": faiss_index.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    """


@cl.on_settings_update
async def setup_agent(settings):
    old_settings = cl.user_session.get("old_settings")
    changes_made = False

    for key in ["formation_lvl", "domaine"]:
        if settings[key] != old_settings[key]:
            changes_made = True
            cl.user_session.set(key, settings[key])

    if changes_made:
        cl.user_session.set(
            "corps_prompt",
            generate_specific_message(settings["domaine"], settings["formation_lvl"]),
        )
        new_old_settings = {
            key: settings[key] for key in old_settings.keys()
        }  # type : dict
        cl.user_session.set("old_settings", new_old_settings)


def generate_specific_message(domaine: Optional[str], formation: Optional[str]) -> str:
    if domaine and formation:
        return f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}. L'utilisateur sort de la formation {formation}."
    elif domaine:
        return f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}."
    elif formation:
        return f"L'utilisateur sort de la formation {formation}."
    return prompt_v3_no_context


@cl.on_message
async def on_message(message: cl.Message):
    ensemble_manager = cl.user_session.get(
        "ensemble_manager"
    )  # type: EnsembleModelManager
    runnable = cl.user_session.get("runnable")
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    msg = await ensemble_manager.stream_ensemble_response(message, runnable)

    # Update memory
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Initialize EnsembleModelManager
    ensemble_manager = EnsembleModelManager()
    cl.user_session.set("ensemble_manager", ensemble_manager)

    memory = ConversationBufferMemory(return_messages=True)

    # Restore chat history
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
            TextInput(id="domaine", label="Domaine d'études de recherche"),
        ]
    ).send()

    # Initialize with default settings
    old_settings = {
        "formation_lvl": None,
        "domaine": None,
    }
    cl.user_session.set("old_settings", old_settings)
