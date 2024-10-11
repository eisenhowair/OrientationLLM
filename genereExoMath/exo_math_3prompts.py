from langchain_community.llms import Ollama
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from chainlit.input_widget import Slider
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict

from prep_message_chainlit_prompts import *



@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("elias", "elias"):
        return cl.User(
            identifier="Elias", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    await cl.ChatSettings(
        [
            Slider(
                id="age_cible",
                label="Age niveau exercice",
                initial=5,
                min=3,
                max=22,
                step=1,
                tooltip="en années",
            ),
        ]
    ).send()

    loisirs = await cl.AskUserMessage(content="Quels sont vos centres d'intérêt?", author="Aide", timeout=3000).send()
    # print(loisirs["output"])
    cl.user_session.set("loisirs", loisirs["output"])
    response = "Merci! Quel genre d'exercice voulez-vous?"
    await cl.Message(content=response, author="Aide").send()

    cl.user_session.set("compris", True)
    cl.user_session.set("tentatives", 0)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))




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
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    if cl.user_session.get("age_niveau"):
        niveau_scolaire = str(cl.user_session.get("age_niveau"))+" ans"
    else:
        niveau_scolaire = "5 ans"

    if cl.user_session.get("compris") == True:  # partie génération d'exercice
        dernier_exo = ""
        print("partie génération d'exercice")
        runnable = setup_exercice_model()

        msg = cl.Message(content="", author="Générateur")
        async for chunk in runnable.astream(
            {"question": message.content, "niveau_scolaire": niveau_scolaire},
            config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
            dernier_exo += chunk
        cl.user_session.set("dernier_exo", dernier_exo)
        cl.user_session.set("compris", False)
        await msg.send()

        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(msg.content)

        # partie correction d'exercice
    elif cl.user_session.get("compris") == False:
        runnable = setup_corrige_model()

        msg = cl.Message(content="", author="Correcteur")
        async for chunk in runnable.astream(
            {"question": message.content,
                "dernier_exo": cl.user_session.get("dernier_exo"),
                "niveau_scolaire": niveau_scolaire
             },
            config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        print("msg:"+str(msg.content))
        await msg.send()
        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(msg.content)
        await verifie_comprehension()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("age_niveau", settings['age_cible'])


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
            Slider(
                id="age_cible",
                label="Age niveau exercice",
                initial=8,
                min=3,
                max=22,
                step=1,
                tooltip="en années",
            ),
        ]
    ).send()

    setup_exercice_model()
