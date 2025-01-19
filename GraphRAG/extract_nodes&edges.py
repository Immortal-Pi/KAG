from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import re
import dotenv 
from langchain_openai import AzureChatOpenAI
# from langchain_openai import AzureOpenAI
from openai import AzureOpenAI
import os  
import json 






url_list=[  
    'https://en.wikipedia.org/wiki/Elon_Musk',
    'https://en.wikipedia.org/wiki/Mark_Zuckerberg',
    'https://en.wikipedia.org/wiki/Bill_Gates',
    'https://en.wikipedia.org/wiki/Jeff_Bezos',
    'https://en.wikipedia.org/wiki/Steve_Jobs',
    'https://en.wikipedia.org/wiki/Sam_Altman',
    'https://en.wikipedia.org/wiki/Larry_Ellison',
    'https://en.wikipedia.org/wiki/Larry_Page',
    'https://en.wikipedia.org/wiki/Sundar_Pichai',
    'https://en.wikipedia.org/wiki/Satya_Nadella' 
    
]
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_data_from_URL(url):
    loader=WebBaseLoader([url])
    data=loader.load().pop().page_content
    data=clean_text(data)
    documents=[Document(page_content=data)]
    # print(documents)
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    smaller_doc=splitter.split_documents(documents)
    print(len(smaller_doc))
    return smaller_doc

dotenv.load_dotenv()
os.environ["OPENAI_API_BASE"] = "https://llmops-amruth.openai.azure.com/"
os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-4"
os.environ["OPENAI_API_VERSION"] = os.getenv('AZURE_OpenAI_API_VERSION')
#os.environ["OPENAI_API_KEY"] = os.getenv('GRAPHRAG_API_KEY')
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('GRAPHRAG_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
# llm=AzureOpenAI(
#         api_key=os.getenv('GRAPHRAG_API_KEY'),
#         azure_endpoint='https://llmops-amruth.openai.azure.com/',
#         api_version=os.getenv('AZURE_OpenAI_API_VERSION')
# )

llm = AzureChatOpenAI(
    deployment_name="gpt-4",
    api_key=os.getenv('GRAPHRAG_API_KEY'),
    azure_endpoint="https://llmops-amruth.openai.azure.com/",
    api_version=os.getenv('AZURE_OpenAI_API_VERSION')
)
system=""" You are a network graph maker tasked with analyzing the relationships involving top leaders in the world. Your job is to process the provided context chunk 
and extract an ontology of terms that represent key entrepreneurs, their associated entities, and all kinds of relationships present in the context.

**Guidelines for Extraction:**

1. **Identify Key Entrepreneurs and Related Terms**:
   - Extract key entrepreneurs and related concepts such as:
     - Companies, organizations, or industries they are associated with.
     - Collaborators, partners, rivals, or competitors.
     - Key innovations, achievements, or milestones.
     - Locations, events, or time periods relevant to their actions.

2. **Identify Relationships**:
   - Extract all types of relationships between entrepreneurs and other entities (or between entities themselves).
   - Relationships can include:
     - Professional roles or associations.
     - Business partnerships, collaborations, or rivalries.
     - Innovations or contributions to industries.
     - Personal connections or influences.
     - Historical events or shared milestones.

3. **Define Relationships**:
   - Clearly specify the nature of each relationship in simple and concise terms.
   - Relationships should convey meaningful connections relevant to the context.

**Response Format**:
- Provide your output **strictly as a list of JSON objects**. No additional text, descriptions,tags or comments are allowed.
- Each object should include the following fields:
  - `"node_1"`: The first entity in the relationship (can be a person, organization, or concept).
  - `"node_2"`: The second entity in the relationship.
  - `"edge"`: A concise sentence describing the relationship between `node_1` and `node_2`.

**Example Output**:
[
   {
       "node_1": "Elon Musk",
       "node_2": "SpaceX",
       "edge": "Elon Musk founded SpaceX to revolutionize space exploration."
   },
   {
       "node_1": "Steve Jobs",
       "node_2": "Apple Inc.",
       "edge": "Steve Jobs co-founded Apple Inc., a leading tech company."
   },
   {
       "node_1": "Mark Zuckerberg",
       "node_2": "Sheryl Sandberg",
       "edge": "Sheryl Sandberg worked closely with Mark Zuckerberg as COO of Facebook."
   },
   {
       "node_1": "Jeff Bezos",
       "node_2": "Blue Origin",
       "edge": "Jeff Bezos founded Blue Origin to focus on space exploration."
   }
]

**Important Note**:
- Always respond exclusively in JSON format. Any deviation from the JSON structure or inclusion of additional text will not be accepted.
- Do not use code block formatting like ` ``` `.
- Output must be a valid JSON array of objects without any surrounding text.

Please provide the context containing information about entrepreneurs and their relationships for analysis.

""" 


from datetime import datetime
dotenv.load_dotenv()
results = []

start_time = datetime.now()
for url in url_list:
    smaller_doc = extract_data_from_URL(url)
    for doc in smaller_doc[:25]:
        try:
            response = llm.predict_messages([
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": doc.page_content,
                }
            ])
            results.append(response.content)
        
        except Exception as e:
            print(e)

end_time = datetime.now()
len(results)
print(f'extracted information in {end_time-start_time}')




combined_nodes_and_edges=[]
for res in results:
    try:
        combined_nodes_and_edges.extend(json.loads(res)) #convert the string result from LLM to JSON 
    except Exception as e:
        print('buggy JSON object', e)

with open('Nodes_and_edges.json','w') as file:
    json.dump(combined_nodes_and_edges,file,indent=1)