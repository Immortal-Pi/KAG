import neo4j
from neo4j_graphrag.llm import AzureOpenAILLM  
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG 
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from dotenv import load_dotenv
import os 
import time 
import asyncio


neo4j_driver=neo4j.GraphDatabase.driver(os.getenv('NEO4J_URI_ONLINE'),auth=(os.getenv('NEO4J_USERNAME_ONLINE'),os.getenv('NEO4J_PASSWORD_ONLINE')))
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



async def process_files(kg_builder_pdf, files):
    files=os.listdir('input')
    for file in files:
        try:
            print(f'Processing: {file}')
            with open(os.path.join('input', file), 'r') as f:
                text_data = f.read()

            # Use only first 15,000 characters to avoid token limit issues
            pdf_results = await kg_builder_pdf.run_async(text=text_data[:15000])
            print(f'Result: {pdf_results}')
        
        except Exception as e:
            errordata = str(e.args[0])
            print(errordata)
            if 'rate limit' in errordata.lower():
                print('Rate limit hit. Sleeping for a minute...')
                await asyncio.sleep(60)

if __name__=='__main__':
    # Define basic node labels
    basic_node_labels = ["Person", "Company", "Startup", "Investor", "VentureCapitalFirm", "FundingRound"]

    # Additional business-related node labels
    business_node_labels = ["Acquisition", "Merger", "Product", "Service", "Market", "Industry", "Technology"]

    # Academic-related node labels (if applicable to research)
    academic_node_labels = ["Publication", "Patent", "Conference", "ResearchPaper"]

    # Funding and financial-related node labels
    finance_node_labels = ["IPO", "Equity", "Grant", "DebtFinancing", "AngelInvestment"]

    # Combine all node labels
    node_labels = basic_node_labels + business_node_labels + academic_node_labels + finance_node_labels

    # Define relationship types
    rel_types = [
        "FOUNDED", "INVESTED_IN", "ACQUIRED", "MERGED_WITH", "PARTNERED_WITH", 
        "EMPLOYED_AT", "ADVISOR_TO", "BOARD_MEMBER_OF", "OWNS", "FUNDED",
        "LICENSED", "HOLDS_PATENT", "PUBLISHED", "COFOUNDED", "WORKS_WITH",
        "SPONSORED", "ACCELERATED_BY", "INCUBATED_BY", "RAISED_FUNDS_IN",
        "PRODUCT_LAUNCHED", "MARKETED_BY", "PIVOTED_TO", "DEVELOPED"
    ]

    prompt_template = '''
    You are a network builder tasked with extracting information from business-related documents 
    and structuring it in a property graph to inform further networking and investment analysis.

    Extract the entities (nodes) and specify their type from the following input text.
    Also, extract the relationships between these nodes. The relationship direction goes from the start node to the end node.

    Return the result as JSON using the following format:
    {{
    "nodes": [ 
        {{"id": "entity_name", "label": "type_of_entity", "properties": {{"name": "name_of_entity"}} }}
    ],
    "relationships": [
        {{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "entity_name_1", "end_node_id": "entity_name_2", 
        "properties": {{"details": "Description of the relationship"}} 
        }} 
    ] 
    }}

    Guidelines:
    - Use only the information from the input text. Do not add any additional information.  
    - If the input text is empty, return an empty JSON. 
    - Extract as many nodes and relationships as needed to offer a rich entrepreneurial and business context for further networking and investment analysis.
    - The property graph should enable an AI knowledge assistant to understand the business context and assist in investment decisions, startup connections, and entrepreneurial insights.
    - Multiple documents will be ingested from different sources, and we are using this property graph to connect information. Ensure entity types remain general and widely applicable.

    Use only the following nodes and relationships (if provided):
    {schema}

    Assign a unique ID (string) to each node and reuse it to define relationships.
    Ensure that relationships respect the source and target node types and follow the correct direction.

    Return **only** the JSON in the specified format—no additional information.

    Examples:
    {examples}

    Input text:

    {text}
    '''
    kg_builder_pdf=SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    text_splitter=FixedSizeSplitter(chunk_size=1000,chunk_overlap=250),
    embedder=embeddings,
    entities=node_labels,
    relations=rel_types,
    prompt_template=prompt_template,
    from_pdf=False,
    on_error='RAISE'
    )
    path='input'
    files=os.listdir(path)

    # print(files)

    

# Call the async function

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_files(kg_builder_pdf, files))
    finally:
        loop.close()