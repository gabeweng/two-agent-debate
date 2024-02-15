import streamlit as st

from typing import Callable, List

from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from langchain.agents import AgentType, initialize_agent, load_tools


# Title
st.title('Two Agent Debate')
