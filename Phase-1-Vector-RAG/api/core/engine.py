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
        """Step 1: Retrieve context if it exists."""
        try:
            # Check if database has data
            if self.db._collection.count() == 0:
                return {"context": "NO_DOCUMENTS_UPLOADED", "is_legal": False}
                
            docs = self.db.similarity_search(state["question"], k=6)
            if not docs:
                return {"context": "NO_RELEVANT_CONTEXT", "is_legal": False}

            context_list = []
            for d in docs:
                source_file = d.metadata.get("source", "Unknown Document")
                page_num = d.metadata.get("page", "N/A")
                if isinstance(page_num, int): page_num += 1 
                
                context_list.append(f"[SOURCE: {source_file}, PAGE: {page_num}]\nCONTENT: {d.page_content}")
                
            return {"context": "\n\n---\n\n".join(context_list)}
        except:
            return {"context": "NO_DOCUMENTS_UPLOADED", "is_legal": False}

    def validate_intent(self, state: AgentState):
        """Step 2: Validate if the question is legal regardless of document presence."""
        validation_prompt = [
            SystemMessage(content=(
                "You are a legal intent classifier. Analyze the user's question and the snippets. "
                "Determine if the user is asking a legal question OR asking about a legal document. "
                "Respond with exactly one word: 'LEGAL' if the question is about law, regulations, "
                "rights, or legal analysis (even if no document is present), 'OTHER' otherwise."
            )),
            HumanMessage(content=f"USER QUESTION: {state['question']}\n\nCONTEXT SNIPPETS: {state['context'][:1000]}")
        ]
        
        response = self.llm.invoke(validation_prompt)
        decision = response.content.strip().upper()
        return {"is_legal": "LEGAL" in decision}

    def out_of_scope_response(self, state: AgentState):
        """Node for strictly non-legal queries (e.g., 'how to bake a cake')"""
        return {
            "answer": "I am a specialized Legal Advisory Agent. I can only assist with legal queries or document analysis. Please ask a law-related question.",
            "history": [HumanMessage(content=state["question"])]
        }

    def generate(self, state: AgentState):
        """Step 3: Generate answer using Context (if relevant) OR Internal Knowledge."""
        ctx = state["context"]
        
        system_prompt_content = (
            "You are a professional Legal Advisory AI. \n\n"
            "GUIDELINES:\n"
            "1. If relevant context is provided, use it and ALWAYS cite using the [SOURCE: filename, PAGE: number] format.\n"
            "2. If no context is provided, or the context is not relevant to the question, "
            "answer the question using your internal legal knowledge.\n"
            "3. If answering from internal knowledge, provide a general legal disclaimer "
            "stating that this is general legal information.\n"
            "4. Use Markdown for professional formatting."
        )

        messages = [SystemMessage(content=system_prompt_content)] + state.get("history", [])
        messages.append(HumanMessage(content=f"CONTEXT PROVIDED:\n{ctx}\n\nUSER QUESTION: {state['question']}"))
        
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "history": [HumanMessage(content=state["question"]), response]
        }

    def intent_router(self, state: AgentState) -> Literal["generate", "out_of_scope"]:
        return "generate" if state["is_legal"] else "out_of_scope"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("validate_intent", self.validate_intent)
        workflow.add_node("generate", self.generate)
        workflow.add_node("out_of_scope", self.out_of_scope_response)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "validate_intent")
        workflow.add_conditional_edges(
            "validate_intent",
            self.intent_router,
            {"generate": "generate", "out_of_scope": "out_of_scope"}
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("out_of_scope", END)
        return workflow.compile()
