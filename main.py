# %% [markdown]
# # Building a LangChain-Based Intelligent Retrieval System with Neo4j and OpenAI

# %%
# Install necessary packages
%pip install --upgrade --quiet langchain langchain-community langchain-openai langchain-experimental neo4j wikipedia yfiles_jupyter_graphs

# %%
# Import required libraries and modules
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from google.colab import userdata
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from yfiles_jupyter_graphs import GraphWidget
from neo4j import GraphDatabase
import os

# %%
# Enable custom widget manager for Colab if available
try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
except:
    pass

# %%
# Import components from LangChain Community
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings

# %%
# Set up environment variables and credentials
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
NEO4J_URI = "**"
NEO4J_USERNAME = "**"
NEO4J_PASSWORD = "**"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

# %%
# Instantiate Neo4jGraph for graph database operations
graph_db = Neo4jGraph()


# %%
# Load example research paper content from Wikipedia
raw_docs = WikipediaLoader(query="Artificial intelligence").load()

# %%
# Split documents into smaller chunks
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
split_docs = text_splitter.split_documents(raw_docs[:3])

# %%
# Initialize ChatOpenAI instance for language model interaction
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

# %%
# Initialize LLMGraphTransformer for converting documents to graph format
llm_graph_transformer = LLMGraphTransformer(llm=llm)

# %%
# Convert documents to graph format
graph_docs = llm_graph_transformer.convert_to_graph_documents(split_docs)

# %%
# Add graph documents to the Neo4j database
graph_db.add_graph_documents(
    graph_docs,
    baseEntityLabel=True,
    include_source=True
)

# %%
# Define the default Cypher query to display graph data
default_cypher_query = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"

# %%
def display_graph(cypher_query: str = default_cypher_query):
    try:
        import google.colab
        from google.colab import output
        output.enable_custom_widget_manager()
    except:
        pass

    # Create a Neo4j session to run queries
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"],
              os.environ["NEO4J_PASSWORD"]))
    session = driver.session()

    # Display graph using yFiles Jupyter Graphs widget
    widget = GraphWidget(graph=session.run(cypher_query).graph())
    widget.node_label_mapping = 'id'
    display(widget)
    return widget

# %%
# Display the graph
display_graph()

# %%
# Instantiate Neo4jVector for vector store operations
vector_store = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# %%
# Create full-text index query for entity extraction
def generate_full_text_query(input_text: str) -> str:
    query = ""
    words = [el for el in remove_lucene_chars(input_text).split() if el]
    for word in words[:-1]:
        query += f" {word}~2 AND"
    query += f" {words[-1]}~2"
    return query.strip()


# %%
# Extract entities from text using the ChatPromptTemplate
class EntityExtraction(BaseModel):
    """Extracted entity information."""
    names: List[str] = Field(
        ...,
        description="List of person, organization, or business entities in the text",
    )

entity_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You will extract organization and person entities from the text."),
        ("human", "Extract the following information from the input text: {question}"),
    ]
)

# %%
# Set up the entity extraction chain using LangChain
entity_extraction_chain = entity_prompt | llm.with_structured_output(EntityExtraction)

# %%
# Perform structured data retrieval using the Neo4j database
def structured_data_retriever(question: str) -> str:
    result = ""
    entities = entity_extraction_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph_db.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# %%
# Test structured data retrieval
print(structured_data_retriever("What is data science?"))

# %%
# Retrieve data combining structured and unstructured sources
def retrieve_data(question: str):
    print(f"Search query: {question}")
    structured_info = structured_data_retriever(question)
    unstructured_info = [doc.page_content for doc in vector_store.similarity_search(question)]
    combined_info = f"""Structured data:
{structured_info}
Unstructured data:
{"#Document ".join(unstructured_info)}
    """
    return combined_info

# %%
# Define a prompt template for condensing conversation history and follow-up questions
condense_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

# %%
# Create a prompt template from the condense template
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

# %%
# Format chat history from a list of tuples to a list of messages
def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# %%
# Define a branch to handle the search query, condense question, and answer the question
search_query_branch = RunnableBranch(
    # If input includes chat_history, condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, no chat history, just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

# %%
# Define a prompt template for answering questions based on context
answer_template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

# %%
# Create a prompt template from the answer template
answer_prompt = ChatPromptTemplate.from_template(answer_template)

# %%
# Define a chain to handle the structured query and return the data
qa_chain = (
    RunnableParallel(
        {
            "context": search_query_branch
            | StrOutputParser()
        }
        | answer_prompt,
    )
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

# %%
# Test the QA chain with a question
print(qa_chain.invoke({"question": "What is artificial intelligence?"}))

# %%
# Save the prompt template and query component
CONDENSE_QUESTION_PROMPT.save("condense_question_prompt.json")
qa_chain.save("qa_chain.json")


