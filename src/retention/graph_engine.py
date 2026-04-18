from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from .vector_store import _get_api_key, store_manager
import pandas as pd
import os

class AgentState(TypedDict):
    customer_profile: Dict
    churn_probability: float
    retrieved_context: Optional[str]
    retention_strategy: Optional[str]

# Define LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=_get_api_key()
)

def retrieve_policies_node(state: AgentState):
    """Retrieves relevant policies based on customer's highest risks."""
    # Convert profile to string to search the vector DB
    profile_str = " ".join([f"{k}: {v}" for k, v in state["customer_profile"].items()])
    
    retriever = store_manager.get_retriever(k=2)
    docs = retriever.invoke(profile_str)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def generate_strategy_node(state: AgentState):
    """Generates the strategy using LLM and retrieved RAG context."""
    template = """
    You are an expert customer retention specialist for a Telecom company.
    
    Customer Profile:
    {profile}
    
    Current Churn Risk: {risk}%
    
    Company Retention Policies and Guidelines:
    {context}
    
    Generate a short, actionable retention strategy tailored specifically to this customer. 
    Point out exactly which policy you are applying. 
    Format your response in Markdown with bullet points.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["profile", "risk", "context"]
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "profile": str(state["customer_profile"]),
        "risk": round(state["churn_probability"] * 100, 2),
        "context": state["retrieved_context"]
    })
    
    return {"retention_strategy": response.content}

# Set up the Graph
graph_builder = StateGraph(AgentState)

# Add Nodes
graph_builder.add_node("retrieve", retrieve_policies_node)
graph_builder.add_node("generate", generate_strategy_node)

# Add Edges
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# Compile Graph
retention_engine = graph_builder.compile()

def run_retention_engine(customer_data: pd.DataFrame, churn_prob: float) -> str:
    """Wrapper function to execute the LangGraph engine."""
    profile_dict = customer_data.iloc[0].to_dict()
    
    initial_state = {
        "customer_profile": profile_dict,
        "churn_probability": churn_prob,
        "retrieved_context": None,
        "retention_strategy": None
    }
    
    result = retention_engine.invoke(initial_state)
    return result["retention_strategy"]
