import gradio as gr
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key  = os.environ['OPENAI_API_KEY']
llm = OpenAI(temperature=0)

def summarize_pdf(pdf_file_path, custom_prompt=""):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

def main():
    input_pdf_path = gr.inputs.Textbox(label="Enter the PDF file path")
    output_summary = gr.outputs.Textbox(label="Summary")

    iface = gr.Interface(
        fn=summarize_pdf,
        inputs=input_pdf_path,
        outputs=output_summary,
        title="PDF Summarizer",
        description="Enter the path to a PDF file and get its summary.",
    )

    iface.launch()

if __name__ == "__main__":
    main()
