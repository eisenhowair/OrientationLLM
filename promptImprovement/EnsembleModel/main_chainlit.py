from langchain_ollama.llms import OllamaLLM
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chainlit.input_widget import TextInput, Select
from prompt_warehouse import *
from prepare_prompt import *

import chainlit as cl
from chainlit.types import ThreadDict
from langchain.memory import ConversationBufferMemory
from ensemble_model_gestion import EnsembleModelManager
from typing import List, Dict, Optional

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
    # Initialize EnsembleModelManager
    ensemble_manager = EnsembleModelManager()
    cl.user_session.set("ensemble_manager", ensemble_manager)

    # Updated settings to reflect available models from EnsembleModelManager
    available_models = list(ensemble_manager.available_models.keys())

    settings = await cl.ChatSettings(
        [
            TextInput(id="formation_lvl", label="Post-"),
            TextInput(id="domaine", label="Domaine d'études de recherche"),
            Select(
                id="use_ensemble",
                label="Mode de fonctionnement",
                values=["Single Model", "Ensemble Model"],
                initial_index=1,
            ),
            Select(
                id="model_choice",
                label="Modèle unique (si non ensemble)",
                values=available_models,
                initial_index=0,
            ),
        ]
    ).send()

    old_settings = {
        "formation_lvl": None,
        "domaine": None,
        "use_ensemble": None,
        "model_choice": None,
    }

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("old_settings", old_settings)


@cl.on_settings_update
async def setup_agent(settings):
    old_settings = cl.user_session.get("old_settings")
    changes_made = False

    for key in ["formation_lvl", "domaine", "use_ensemble", "model_choice"]:
        if settings[key] != old_settings[key]:
            changes_made = True
            cl.user_session.set(key, settings[key])

    if changes_made:
        cl.user_session.set(
            "corps_prompt",
            generate_specific_message(settings["domaine"], settings["formation_lvl"]),
        )
        new_old_settings = {key: settings[key] for key in old_settings.keys()}
        cl.user_session.set("old_settings", new_old_settings)

        ensemble_manager = cl.user_session.get(
            "ensemble_manager"
        )  # type: EnsembleModelManager

        if settings["use_ensemble"] == "Ensemble Model":
            # Activate all available models for ensemble
            ensemble_manager.activate_models(
                list(ensemble_manager.available_models.keys())
            )
            runnable = ensemble_manager.create_ensemble_runnable(
                domaine=settings["domaine"], formation=settings["formation_lvl"]
            )
            cl.user_session.set("runnable", runnable)
        else:
            # Single model mode (using existing logic)
            setup_single_model(
                settings["domaine"], settings["formation_lvl"], settings["model_choice"]
            )


def setup_single_model(
    domaine: Optional[str], formation: Optional[str], model_name: str
):
    model = OllamaLLM(
        base_url="http://localhost:11434",
        model=model_name,
        cache=True,
        num_ctx=32768,
        repeat_penalty=1.3,
    )
    specific_message = generate_specific_message(domaine, formation)
    cl.user_session.set("corps_prompt", specific_message)
    runnable = prepare_prompt_few_shot(model=model)
    cl.user_session.set("runnable", runnable)


def generate_specific_message(domaine: Optional[str], formation: Optional[str]) -> str:
    if domaine and formation:
        return f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}. L'utilisateur sort de la formation {formation}."
    elif domaine:
        return f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}."
    elif formation:
        return f"L'utilisateur sort de la formation {formation}."
    return prompt_no_domain_no_formation_v3


@cl.on_message
async def on_message(message: cl.Message):
    ensemble_manager = cl.user_session.get("ensemble_manager")
    runnable = cl.user_session.get("runnable")
    memory = cl.user_session.get("memory")
    use_ensemble = cl.user_session.get("use_ensemble")

    if use_ensemble == "Ensemble Model":
        await ensemble_manager.stream_ensemble_response(message, runnable)
    else:
        msg = cl.Message(content="")
        async for chunk in runnable.astream(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        await msg.send()

    # Update memory
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(
        msg.content if "msg" in locals() else "Response sent via ensemble"
    )


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

    # Restore settings with available models from EnsembleModelManager
    available_models = list(ensemble_manager.available_models.keys())

    settings = await cl.ChatSettings(
        [
            TextInput(id="formation_lvl", label="Post-"),
            TextInput(id="domaine", label="Domaine d'études de recherche"),
            Select(
                id="use_ensemble",
                label="Mode de fonctionnement",
                values=["Single Model", "Ensemble Model"],
                initial_index=0,
            ),
            Select(
                id="model_choice",
                label="Modèle unique (si non ensemble)",
                values=available_models,
                initial_index=0,
            ),
        ]
    ).send()

    # Initialize with default settings
    old_settings = {
        "formation_lvl": None,
        "domaine": None,
        "use_ensemble": None,
        "model_choice": None,
    }
    cl.user_session.set("old_settings", old_settings)