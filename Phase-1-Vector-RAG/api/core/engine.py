import os
import operator
from typing import TypedDict, List, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from api.core.database import get_vector_store

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    is_legal: bool
    history: Annotated[List[BaseMessage], operator.add]

class LegalEngine:
    def __init__(self):
        self.db = get_vector_store()
        self.llm = ChatOpenRouter(
            model="openai/gpt-oss-120b:free",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1
        )

    def retrieve(self, state: AgentState):
        """
        Step 1: Always retrieve context first so we know what the document is about.
        """
        docs = self.db.similarity_search(state["question"], k=6)
        
        # Check if database is empty
        if not docs:
            return {"context": "EMPTY_DB", "is_legal": False}

        context_list = []
        for d in docs:
            source_file = d.metadata.get("source", "Unknown Document")
            context_list.append(f"[SOURCE: {source_file}]\nCONTENT: {d.page_content}")
            
        return {"context": "\n\n---\n\n".join(context_list)}

    def validate_intent(self, state: AgentState):
        """
        Step 2: Check if the QUESTION is legal OR if the DOCUMENT being summarized is legal.
        """
        if state["context"] == "EMPTY_DB":
            return {"is_legal": False, "answer": "No documents found. Please upload a file first."}

        validation_prompt = [
            SystemMessage(content=(
                "You are a legal document classifier. Analyze the user's question and the provided document snippets. "
                "Determine if the user is asking a legal question OR asking to summarize/analyze a legal document. "
                "Legal documents include contracts, NDAs, court papers, statutes, policies, and regulations. "
                "Respond with exactly one word: 'LEGAL' if the context OR question is law-related, 'OTHER' otherwise."
            )),
            HumanMessage(content=f"USER QUESTION: {state['question']}\n\nDOCUMENT SNIPPETS: {state['context'][:2000]}")
        ]
        
        response = self.llm.invoke(validation_prompt)
        decision = response.content.strip().upper()
        
        is_legal = "LEGAL" in decision
        return {"is_legal": is_legal}

    def out_of_scope_response(self, state: AgentState):
        """Node for non-legal or empty-db cases"""
        # If we already set an answer (like 'No documents found'), use it.
        msg = state.get("answer") or "I am a specialized Legal Advisory Agent. This document or question does not appear to be legal in nature."
        return {
            "answer": msg,
            "history": [HumanMessage(content=state["question"])]
        }

    def generate(self, state: AgentState):
        """Step 3: Generate the answer (Summary or Q&A)"""
        system_prompt = SystemMessage(content=(
            "You are a professional Legal Advisory AI. "
            "If the user asks for a summary, provide a concise but comprehensive overview of the key legal points. "
            "If the user asks a specific question, answer it using the context. "
            "ALWAYS cite the [SOURCE: filename]. Use Markdown."
        ))
        
        messages = [system_prompt] + state.get("history", [])
        messages.append(HumanMessage(content=f"CONTEXT:\n{state['context']}\n\nQUESTION: {state['question']}"))
        
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "history": [HumanMessage(content=state["question"]), response]
        }

    def intent_router(self, state: AgentState) -> Literal["generate", "out_of_scope"]:
        """Routes based on the combined validation of doc + question"""
        return "generate" if state["is_legal"] else "out_of_scope"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("validate_intent", self.validate_intent)
        workflow.add_node("generate", self.generate)
        workflow.add_node("out_of_scope", self.out_of_scope_response)
        
        # Workflow: Start -> Retrieve -> Validate -> Route
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "validate_intent")
        
        workflow.add_conditional_edges(
            "validate_intent",
            self.intent_router,
            {
                "generate": "generate",
                "out_of_scope": "out_of_scope"
            }
        )
        
        workflow.add_edge("generate", END)
        workflow.add_edge("out_of_scope", END)
        
        return workflow.compile()