import streamlit as st
from neo4j_graphrag.indexes import create_vector_index
import neo4j
import os 
import neo4j
from neo4j_graphrag.llm import AzureOpenAILLM  
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG 
from dotenv import load_dotenv
import os 
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import RagTemplate







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

st.set_page_config(page_title="Multi- Response Viewer", layout="wide")

st.title("💬 Graph RAG - question")

# User question input at the top
user_question = st.text_input("Type your question here:", placeholder="e.g., How does AI affect healthcare?")

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
        st.write(f"Vector response: {vc_rag.search(user_question,retriever_config={'top_k':10}).answer}")


    with col2:
        st.subheader("Microsoft-GraphRAG")
        st.write("Response from Model B based on the question:")
        st.info(f"🔸 Another perspective on: '{user_question}'")

    with col3:
        st.subheader("GraphQAChain")
        st.write("Response from Model C based on the question:")
        st.warning(f"🟡 A different take on: '{user_question}'")
