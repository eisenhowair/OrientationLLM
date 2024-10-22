from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from prompt_warehouse import *
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, Runnable
from operator import itemgetter
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable.config import RunnableConfig

from chainlit.types import ThreadDict


def prepare_prompt_few_shot(corps_prompt, model):

    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    shots = [
        {"input": FS_human_example_1, "output": FS_model_example_1},
        {"input": FS_human_example_2, "output": FS_model_example_2},
        {"input": FS_human_example_3, "output": FS_model_example_3},
    ]

    example_prompt = HumanMessagePromptTemplate.from_template(
        "{input}"
    ) + AIMessagePromptTemplate.from_template("{output}")

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=shots,
        example_prompt=example_prompt,
    )
    """
    # Construire le prompt
    # prompt_orientation = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", f"{corps_prompt}"),
    #         MessagesPlaceholder(variable_name="history")
    #         #("human", "{question}"),
    #     ]
    # )
    """

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
