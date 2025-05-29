import streamlit as st
import neo4j
from neo4j_graphrag.indexes import create_vector_index
import os 
from neo4j_graphrag.llm import AzureOpenAILLM  
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG 
from dotenv import load_dotenv

from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import RagTemplate

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain_openai import AzureChatOpenAI
import dotenv
from pathlib import Path 
import json 
import pandas as pd 
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END 
import subprocess
from pathlib import Path 
project_root=Path('D:/pythonProjects/KAG_Testing')


# neo4j 
llm=AzureOpenAILLM(
    model_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL_2'),
    azure_endpoint=os.getenv('AZURE_OpenAI_ENDPOINT_2'),
    api_version=os.getenv('AZURE_OpenAI_API_VERSION_2'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY_2'),
    
)
embeddings=AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT_2'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY_2'),
    api_version=os.getenv('AZURE_OpenAI_API_VERSION_2')
)

neo4j_driver=neo4j.GraphDatabase.driver(os.getenv('NEO4J_URI_ONLINE'),auth=(os.getenv('NEO4J_USERNAME_ONLINE'),os.getenv('NEO4J_PASSWORD_ONLINE')))

create_vector_index(neo4j_driver, name='text_embeddings',label='Chunk', embedding_property='embedding',dimensions=1536, similarity_fn='cosine')

# langQA chain 
llm_qa=AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OpenAI_API_VERSION"),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL'),
    model_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL'),
    api_key=os.getenv('GRAPHRAG_API_KEY'),
   )
with open(Path('Nodes_and_edges.json'),'r') as f:
    data=json.load(f)

graph=NetworkxEntityGraph()

# add nodes to the graph
for node in data:
    try:
        graph.add_node(node['node_1'])
        graph.add_node(node['node_2'])
    except Exception as e:
        print(f'node not found {e}')

# add edges to the graph 
for edge in data:
    try:
        graph._graph.add_edge(
            edge['node_1'],
            edge['node_2'],
            relation=edge['edge']
        ) 
    except Exception as e:
        print(e)

rag_chain=GraphQAChain.from_llm(
    llm=llm_qa,
    graph=graph,
    verbose=True
)
class GraphState(TypedDict):
    question: Optional[str]=None 
    classificaiton: Optional[str]=None
    response: Optional[str]=None 
    greeting: Optional[str]=None 
    revised_question: Optional[str]=None 

def classify(question):
    return llm_qa(f'classify intent of given input as greeting or not_greeting. Output just the class "greeting" or "not_greeting".Input :{question}')

def classify_input_node(state):
    question=state.get('question','').strip()
    classification=classify(question)
    return {'Classification':classification}

def handle_greeting_node(state):
    return {'greeting':'Hello There'}

def handle_RAG(state):
    question=state.get('revised_question','').strip()
    prompt=question
    search_result=rag_chain.run(prompt)
    question=llm_qa(f'Rephrase this questions: {question}')
    print('question',question)
    print('response',search_result)
    return {'response':search_result,'revised_question':question}

def bye(state):
    return {'greeting':"the graph has finished"}

workflow=StateGraph(GraphState)
workflow.add_node('classify_input',classify_input_node)
workflow.add_node('handle_greeting',handle_greeting_node)
workflow.add_node('handle_rag',handle_RAG)
workflow.add_node('bye',bye)

workflow.set_entry_point('classify_input')
workflow.add_edge('handle_greeting',END)
workflow.add_edge('bye',END)

def decide_next_node(state):
    return 'handle_greeting' if state.get('classification')=='greeting' else 'handle_rag'

def check_RAG_length(state):
    response=state.get('response')
    question=state.get('question')
    meaning=llm_qa(f'Response:{response}, Does the response means: "I cannot answer the question?" Output Yes or No')
    print('is the output I dont know??',meaning)
    return 'bye' if 'no' in str(meaning).lower() else 'handle_rag'

workflow.add_conditional_edges(
    'classify_input',
    decide_next_node,
    {
        'handle_greeting':'handle_greeting',
        'handle_rag':'handle_rag'
    }
)
workflow.add_conditional_edges(
    'handle_rag',
    check_RAG_length,
    {
        'bye':'bye',
        'handle_rag':'handle_rag'
    }
)

qachain=workflow.compile()




st.set_page_config(page_title="Multi- Response Viewer", layout="wide")

st.title("KnowGraph Lab")

# User question input at the top
user_question = st.text_input("Type your question here:", placeholder="e.g., who is Jeff Bezos")

# Only show responses if a question is entered
if user_question:
    st.markdown("### 🤖 Responses from Chatbots")

    # Create 3 columns for 3 different responses
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Neo4JGraphRAG")
        
        vc_retriever = VectorCypherRetriever(
        neo4j_driver,
        index_name="text_embeddings",
        embedder=embeddings,
        retrieval_query="""
        //1) Go out 2-3 hops in the entity graph and get relationships
        WITH node AS chunk
        MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
        UNWIND relList AS rel

        //2) collect relationships and text chunks
        WITH collect(DISTINCT chunk) AS chunks,
        collect(DISTINCT rel) AS rels

        //3) format and return context
        RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
        apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
        """
        )
        
        rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

        # Question:
        {query_text}

        # Context:
        {context}

        # Answer:
        ''', expected_inputs=['query_text', 'context'])

        vc_rag = GraphRAG(llm=llm, retriever=vc_retriever, prompt_template=rag_template)

        # st.write("Response from Model A based on the question:")
        # st.success(f"🔹 This is a simulated response to: '{user_question}'")
        st.write(f"{vc_rag.search(user_question,retriever_config={'top_k':3}).answer}")


    with col2:
        st.subheader("Microsoft-GraphRAG")
        try:
            command = [
                "python", "-m", "graphrag", "query",
                "--root", ".", "--method", "global",
                "--query", user_question
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True  # Raises CalledProcessError if the command fails
            )

            print("✅ STDOUT:\n", result.stdout)
            print("ℹ️ STDERR:\n", result.stderr)

            # Store for further use
            output_text = result.stdout
            error_text = result.stderr
            if "SUCCESS: Global Search Response:" in output_text:
                llm_response = output_text.split("SUCCESS: Global Search Response:")[-1].strip()
            else:
                llm_response = "No LLM response found."

            st.write(llm_response)

        except Exception as e:
            print("⚠️ An unexpected error occurred.")
            print(str(e))

    with col3:
        st.subheader("GraphQAChain")
        response=qachain.invoke({'question':user_question,'revised_question':user_question})
        st.write(f"{response['response']}")
        # st.warning(f"🟡 A different take on: '{user_question}'")
