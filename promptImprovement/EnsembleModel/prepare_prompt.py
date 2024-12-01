from langchain_community.llms import Ollama
from typing import List, Dict, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, Runnable
from operator import itemgetter
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable.config import RunnableConfig
from chainlit.types import ThreadDict
from prompt_warehouse import *
from vector_store_manager import *


def prepare_prompt_zero_shot(model):

    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    corps_prompt = cl.user_session.get("corps_prompt")

    prompt_orientation = ChatPromptTemplate.from_messages(
        [
            ("system", f"{corps_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(lambda _: memory.load_memory_variables({}))
            | itemgetter("history")
        )
        | prompt_orientation
        | model
        | StrOutputParser()
    )
    return runnable


def prepare_prompt_few_shot(model):

    corps_prompt = cl.user_session.get("corps_prompt")
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    intro_few_shot = """
    [EXEMPLES DE CONVERSATIONS PASSÉES]
Les exemples suivants illustrent le type d'échange attendu. Utilise-les comme référence pour le ton et la structure, sans les confondre avec les nouvelles questions.
"""
    shots = [
        {"input": FS_human_example_1, "output": FS_model_example_1},
        {"input": FS_human_example_2, "output": FS_model_example_2},
        {"input": FS_human_example_3, "output": FS_model_example_3},
    ]

    example_prompt = (
        SystemMessagePromptTemplate.from_template(
            intro_few_shot
        )  # pour que le modèle arrive à différencier les examples passés de la discussion actuelle
        + HumanMessagePromptTemplate.from_template("{input}")
        + AIMessagePromptTemplate.from_template("{output}")
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=shots,
        example_prompt=example_prompt,
    )

    prompt_orientation = ChatPromptTemplate.from_messages(
        [
            ("system", f"{corps_prompt}"),
            few_shot_prompt,
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # formatted_prompt = prompt_orientation.format(input="Your current question")

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(lambda _: memory.load_memory_variables({}))
            | itemgetter("history")
        )
        | prompt_orientation
        | model
        | StrOutputParser()
    )
    return runnable
