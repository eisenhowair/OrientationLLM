import chainlit as cl
import sys
import os

from dotenv import load_dotenv

from chainlit.input_widget import TextInput, Select
from chainlit.types import ThreadDict
from langchain_ollama.llms import OllamaLLM
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import Runnable
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))


from ..vector_store_manager import VectorStoreFAISS
from ..prompt_warehouse import (
    prompt_v3_context,
    prompt_v3_no_context,
    prompt_v4_context,
    prompt_v4_context_strict,
)
from ..prepare_prompt import prepare_prompt_zero_shot, prepare_prompt_few_shot
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
    load_dotenv()
    ensemble_model_path = os.getenv("VECTORSTORE_INDEX_PATH")
    if not ensemble_model_path:
        raise ValueError(
            "La variable d'environnement VECTORSTORE_INDEX_PATH n'est pas définie."
        )

    vectorstore = VectorStoreFAISS(
        embedding_model_name="hkunlp/instructor-large",
        index_path=ensemble_model_path,
    )  # on réutilise le vectorstore déjà existant dans EnsembleModel
    cl.user_session.set("vectorstore", vectorstore)

    # mise en place du model+prompt (le runnable donc)
    setup_model()


@cl.on_settings_update
async def settings_agent(settings):
    old_settings = cl.user_session.get("old_settings")
    changes_made = False

    for key in ["formation_lvl", "domaine"]:
        if settings[key] != old_settings[key]:
            changes_made = True
            cl.user_session.set(key, settings[key])

    if changes_made:
        setup_model()  # update runnable
        new_old_settings = {
            key: settings[key] for key in old_settings.keys()
        }  # type : dict
        cl.user_session.set("old_settings", new_old_settings)


def setup_model(need_context=False, nom_model=MODEL):
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

    if need_context:
        specific_message = prompt_v4_context_strict
    else:
        specific_message = prompt_v3_no_context

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
    # rag_decider = RAGDecider(response_length=2,model_name="mistralai/Ministral-8B-Instruct-2410",provider = "HF")
    rag_decider = RAGDecider(response_length=2, model_name=MODEL, provider="OL")
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

    need_context = await stream_rag_decider_response(
        message=message, runnable=rag_decider.runnable
    )
    need_context_str = need_context.content  # contient 1 ou 2 en string
    need_context = int(need_context_str)
    print(type(need_context))

    if need_context == 1:
        setup_model(need_context=True)  # pour changer le prompt
        msg = await stream_response(message, runnable, vectorstore=vectorstore)
    else:
        msg = await stream_response(
            message, runnable, vectorstore=None
        )  # type: cl.Message

    # Update memory
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)


@cl.step(name="Multi Agent")
async def stream_rag_decider_response(
    message: cl.Message, runnable: Runnable
) -> cl.Message:
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {
            "input": message.content,
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        #await msg.stream_token(chunk)
        msg.content += chunk
    #await msg.send()
    print(f"Besoin de contexte (1 Oui/2 Non):{msg.content}")
    return msg


async def stream_response(
    message: cl.Message, runnable: Runnable, vectorstore: VectorStoreFAISS
) -> cl.Message:
    msg = cl.Message(content="")
    if vectorstore is not None:
        context = vectorstore.similarity_search(query=message.content)
        for result in context:
            print(f"Contexte récupéré:{result}\n======\n")
    else:
        context = ""

    async for chunk in runnable.astream(
        {
            "input": message.content,
            "context": context,  # rajouté ça pour donner accès au contexte, et rajouté {context} dans le prompt lui-même
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
    setup_model()
    setup_multi_agent()
