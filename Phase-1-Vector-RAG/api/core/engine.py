import os
import operator
from typing import TypedDict, List, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from api.core.database import get_vector_store

# Define the state for the LangGraph workflow
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    is_legal: bool
    # History is annotated with operator.add so messages append automatically
    history: Annotated[List[BaseMessage], operator.add]

class LegalEngine:
    def __init__(self):
        self.db = get_vector_store()
        self.llm = ChatOpenRouter(
            model="openai/gpt-oss-120b:free",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1
        )

    def validate_intent(self, state: AgentState):
        """
        GUARDRAIL: Determines if the query is legal-related.
        This allows general legal questions even without uploaded documents.
        """
        validation_prompt = [
            SystemMessage(content=(
                "You are a legal intent classifier. Your job is to determine if a user's question "
                "is related to law, legal concepts, regulations, or document analysis. "
                "Respond with exactly one word: 'LEGAL' or 'OTHER'. "
                "Examples of LEGAL: 'What is a contract?', 'Summarize my file', 'How does IP law work?'. "
                "Examples of OTHER: 'What is the weather?', 'How to bake a cake?', 'Who won the game?'."
            )),
            HumanMessage(content=state["question"])
        ]
        
        response = self.llm.invoke(validation_prompt)
        decision = response.content.strip().upper()
        return {"is_legal": "LEGAL" in decision}

    def retrieve(self, state: AgentState):
        """
        RETRIEVAL: Fetches context if documents exist. 
        If no documents are indexed, it passes a flag to the generator.
        """
        try:
            # Check if the database has any documents
            # Note: _collection.count() is a standard way to check Chroma status
            if self.db._collection.count() == 0:
                return {"context": "DIRECT_KNOWLEDGE_MODE"}

            docs = self.db.similarity_search(state["question"], k=6)
            
            if not docs:
                return {"context": "NO_RELEVANT_CHUNKS"}

            context_list = []
            for d in docs:
                source_file = d.metadata.get("source", "Unknown Document")
                context_list.append(f"[SOURCE: {source_file}]\nCONTENT: {d.page_content}")
            
            return {"context": "\n\n---\n\n".join(context_list)}
        except Exception:
            # Fallback if DB isn't initialized or accessible
            return {"context": "DIRECT_KNOWLEDGE_MODE"}

    def generate(self, state: AgentState):
        """
        GENERATION: Smart switching between RAG (document-based) and General Advisory.
        """
        ctx = state.get("context", "")
        
        if ctx == "DIRECT_KNOWLEDGE_MODE" or ctx == "NO_RELEVANT_CHUNKS":
            # Mode: General Legal Advice
            system_instruction = (
                "You are a professional Legal Advisory AI. Currently, there is no specific document context "
                "available to answer this question. \n\n"
                "INSTRUCTION: Answer the question using your internal legal knowledge. "
                "Clearly state that your answer is based on general legal principles and not on a specific uploaded document. Use Markdown for formatting."
            )
        else:
            # Mode: Document RAG
            system_instruction = (
                "You are a professional Legal Advisory AI analyzing uploaded documents. \n\n"
                "INSTRUCTION: Use the provided context to answer. ALWAYS cite the [SOURCE: filename] "
                "provided in the context for every claim you make. Use Markdown for formatting."
            )

        messages = [SystemMessage(content=system_instruction)] + state.get("history", [])
        messages.append(HumanMessage(content=f"CONTEXT:\n{ctx}\n\nUSER QUESTION: {state['question']}"))
        
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "history": [HumanMessage(content=state["question"]), response]
        }

    def out_of_scope_response(self, state: AgentState):
        """Refusal for non-legal queries"""
        return {
            "answer": "I am a specialized Legal Advisory Agent. I can only assist with law-related queries or document analysis. Please ask a legal question.",
            "history": [HumanMessage(content=state["question"])]
        }

    def intent_router(self, state: AgentState) -> Literal["retrieve", "out_of_scope"]:
        """Routes based on the validation node results"""
        return "retrieve" if state["is_legal"] else "out_of_scope"

    def build_graph(self):
        """Constructs the LangGraph state machine"""
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("validate_intent", self.validate_intent)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("out_of_scope", self.out_of_scope_response)
        
        # Set Flow
        workflow.set_entry_point("validate_intent")
        
        workflow.add_conditional_edges(
            "validate_intent",
            self.intent_router,
            {
                "retrieve": "retrieve",
                "out_of_scope": "out_of_scope"
            }
        )
        
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        workflow.add_edge("out_of_scope", END)
        
        return workflow.compile()
