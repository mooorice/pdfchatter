from dotenv import load_dotenv
import os
import openai

# Langchain tools
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Gradio
import gradio as gr

# API key
load_dotenv()
openai.api_key  = os.environ['OPENAI_API_KEY']

# PDF path
pdf_path = "./pdf/Currie_traffic_infant_ezpass.pdf"

# Setup LLM
llm = OpenAI(temperature=0, top_p=0.1, n=1)

loader = PyPDFLoader(pdf_path)
docs = loader.load_and_split()
embeddings = OpenAIEmbeddings()
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="Currie_traffic_infant_ezpass",
)

# Setup system prompt

prompt_template = """Use the following pieces of context to answer the question
at the end. If you don't know the answer, just say that you don't know,
don't try to make up an answer. Makes sure to answer the question thoroughly.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Give some user prompt examples

example_questions=[
    "What is the main argument of this paper?",
    "How does the author support the main argument?",
    "What is the author's conclusion?",
]

def get_answers(query, history):
    found_docs = qdrant.max_marginal_relevance_search(query, k=2, fetch_k=10)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(question = query, input_documents = found_docs)
    return summary

def main():
    pdf_chatter = gr.ChatInterface(fn=get_answers, examples=example_questions, title="PDF Chatter")
    pdf_chatter.launch()

if __name__ == "__main__":
    main()
