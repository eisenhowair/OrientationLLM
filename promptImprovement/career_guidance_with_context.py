from langchain_ollama.llms import OllamaLLM
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput, Select
from prompt_warehouse import *
from prepare_prompt import *

import chainlit as cl
from chainlit.types import ThreadDict
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional
from vector_store_manager import *
from RAGDecider import RAGDecider

MODEL = "qwen2.5:3b-instruct"


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

    # mise en place du RAG
    vectorstore = VectorStoreFAISS(
        embedding_model_name="hkunlp/instructor-large",
        index_path="EnsembleModel/embedding_indexes",
    )  # on réutilise le vectorstore déjà existant dans EnsembleModel
    cl.user_session.set("vectorstore", vectorstore)

    # mise en place du model+prompt (le runnable donc)
    setup_model(domaine=None, formation=None)


@cl.on_settings_update
async def settings_agent(settings):
    old_settings = cl.user_session.get("old_settings")
    changes_made = False

    for key in ["formation_lvl", "domaine"]:
        if settings[key] != old_settings[key]:
            changes_made = True
            cl.user_session.set(key, settings[key])

    if changes_made:
        setup_model(
            domaine=settings["domaine"], formation=settings["formation_lvl"]
        )  # update runnable
        new_old_settings = {
            key: settings[key] for key in old_settings.keys()
        }  # type : dict
        cl.user_session.set("old_settings", new_old_settings)


def setup_model(domaine, formation, nom_model=MODEL):
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
        specific_message = prompt_no_domain_no_formation_v3_context

    model = OllamaLLM(
        base_url="http://localhost:11434",
        model=nom_model,
        cache=True,
        num_ctx=32768,
        repeat_penalty=1.3,
    )
    runnable = prepare_prompt_zero_shot(corps_prompt=specific_message, model=model)
    cl.user_session.set("runnable", runnable)
    print("runnable dans la session")
    setup_multi_agent()


def setup_multi_agent():
    rag_decider = RAGDecider()
    rag_decider.prepare_runnable()
    print(f"RAGDecider returned: {type(rag_decider)}")
    print(f"Runnable returned: {type(rag_decider.runnable)}")
    cl.user_session.set("runnable_multi_agent", rag_decider)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    vectorstore = cl.user_session.get("vectorstore")  # type: VectorStoreFAISS

    rag_decider = cl.user_session.get("runnable_multi_agent")  # type: RAGDecider
    need_context = rag_decider.invoke_agent(user_input=message.content)

    msg = await stream_response(message, runnable, vectorstore)  # type: cl.Message

    # Update memory
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)


async def stream_response(
    message: cl.Message, runnable: Runnable, vectorstore: VectorStoreFAISS
) -> cl.Message:
    msg = cl.Message(content="")
    context = vectorstore.similarity_search(query=message.content)
    for result in context:
        print(f"Contexte récupéré:{result}\n======\n")

    async for chunk in runnable.astream(
        {
            "input": message.content,
            "context": context,  # rajouté ça pour donner accès au contexte, et rajouté {context dans le prompt lui-même}
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    return msg


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):

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

    # mise en place du RAG
    vectorstore = VectorStoreFAISS(
        embedding_model_name="hkunlp/instructor-large",
        index_path="EnsembleModel/embedding_indexes",
    )  # on réutilise le vectorstore déjà existant dans EnsembleModel
    cl.user_session.set("vectorstore", vectorstore)

    # mise en place du model+prompt (le runnable donc)
    setup_model(domaine=None, formation=None)
    setup_multi_agent()
