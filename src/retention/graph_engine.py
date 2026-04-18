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
    """Wrapper function to execute the LangGraph engine for a single customer."""
    profile_dict = customer_data.iloc[0].to_dict()

    initial_state = {
        "customer_profile": profile_dict,
        "churn_probability": churn_prob,
        "retrieved_context": None,
        "retention_strategy": None
    }

    result = retention_engine.invoke(initial_state)
    return result["retention_strategy"]


# ---------------------------------------------------------------------------
# Batch / Segment-Level Retention Engine
# ---------------------------------------------------------------------------

class BatchAgentState(TypedDict):
    segment_profile: Dict
    retrieved_context: Optional[str]
    retention_strategy: Optional[str]


def _build_segment_profile(at_risk_df: pd.DataFrame, probas: list) -> Dict:
    """
    Summarise all at-risk customers into a single aggregate profile.
    Returns a dict with population-level statistics used as the LLM prompt.
    """
    n = len(at_risk_df)

    def top(col):
        return at_risk_df[col].mode()[0] if col in at_risk_df.columns else "N/A"

    def pct(col, val):
        if col not in at_risk_df.columns:
            return "N/A"
        return f"{(at_risk_df[col] == val).mean() * 100:.1f}%"

    _SVC = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    avg_services = (
        at_risk_df[_SVC].apply(lambda r: (r == "Yes").sum(), axis=1).mean()
        if all(c in at_risk_df.columns for c in _SVC) else "N/A"
    )

    return {
        "total_at_risk_customers":       n,
        "avg_churn_probability":         f"{sum(probas) / n * 100:.1f}%",
        "avg_tenure_months":             f"{at_risk_df['tenure'].mean():.1f}" if "tenure" in at_risk_df.columns else "N/A",
        "avg_monthly_charges":           f"${at_risk_df['MonthlyCharges'].mean():.2f}" if "MonthlyCharges" in at_risk_df.columns else "N/A",
        "most_common_contract":          top("Contract"),
        "pct_month_to_month":            pct("Contract", "Month-to-month"),
        "most_common_internet_service":  top("InternetService"),
        "pct_fiber_optic":               pct("InternetService", "Fiber optic"),
        "most_common_payment_method":    top("PaymentMethod"),
        "pct_electronic_check":          pct("PaymentMethod", "Electronic check"),
        "pct_senior_citizen":            f"{at_risk_df['SeniorCitizen'].mean() * 100:.1f}%" if "SeniorCitizen" in at_risk_df.columns else "N/A",
        "pct_no_partner":                pct("Partner", "No"),
        "pct_no_dependents":             pct("Dependents", "No"),
        "pct_paperless_billing":         pct("PaperlessBilling", "Yes"),
        "avg_active_services":           f"{avg_services:.1f}" if isinstance(avg_services, float) else avg_services,
    }


def _retrieve_batch_policies(state: BatchAgentState):
    """RAG retrieval using the aggregate segment profile as the query."""
    profile_str = " ".join([f"{k}: {v}" for k, v in state["segment_profile"].items()])
    retriever = store_manager.get_retriever(k=3)
    docs = retriever.invoke(profile_str)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def _generate_batch_strategy(state: BatchAgentState):
    """Generate one cost-effective, segment-wide retention strategy via Gemini."""
    template = """
You are a senior customer retention strategist at a Telecom company.

You have analysed a segment of {n} customers who are predicted to churn.
Below is their aggregate profile — summarised from the entire at-risk population:

{profile}

Company Retention Policies and Guidelines:
{context}

Your task: Design a SINGLE, cost-effective retention programme that addresses the
root causes driving this entire segment to churn. The strategy should:
1. Target the most impactful levers (e.g. contract upgrade incentives, billing changes)
2. Prioritise actions with the highest ROI — retaining even 30% of this group matters
3. Be specific: name exact offers, discounts, service bundles, or policy changes
4. Cite which company policy or guideline you are applying for each recommendation
5. Flag any demographic sub-segments (e.g. senior, no-partner, high-bill) needing different treatment

Format your response in Markdown with clear sections:
## Root Cause Analysis
## Recommended Retention Initiatives
## Estimated Impact
## Sub-Segment Targeting (if applicable)
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["n", "profile", "context"]
    )
    chain = prompt | llm
    profile = state["segment_profile"]
    response = chain.invoke({
        "n":       profile["total_at_risk_customers"],
        "profile": "\n".join([f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in profile.items()]),
        "context": state["retrieved_context"],
    })
    return {"retention_strategy": response.content}


# Build batch StateGraph
_batch_graph_builder = StateGraph(BatchAgentState)
_batch_graph_builder.add_node("retrieve", _retrieve_batch_policies)
_batch_graph_builder.add_node("generate", _generate_batch_strategy)
_batch_graph_builder.add_edge(START, "retrieve")
_batch_graph_builder.add_edge("retrieve", "generate")
_batch_graph_builder.add_edge("generate", END)
_batch_retention_engine = _batch_graph_builder.compile()


def run_batch_retention_engine(at_risk_df: pd.DataFrame, probas: list) -> str:
    """
    Generate ONE collective retention strategy for all at-risk customers.

    Parameters
    ----------
    at_risk_df : DataFrame of customers predicted to churn (raw features, pre-engineered)
    probas     : list of churn probabilities (floats 0–1) for each at-risk customer

    Returns
    -------
    str  — Markdown-formatted retention programme for the whole segment
    """
    segment_profile = _build_segment_profile(at_risk_df, probas)

    result = _batch_retention_engine.invoke({
        "segment_profile":    segment_profile,
        "retrieved_context":  None,
        "retention_strategy": None,
    })
    return result["retention_strategy"], segment_profile
