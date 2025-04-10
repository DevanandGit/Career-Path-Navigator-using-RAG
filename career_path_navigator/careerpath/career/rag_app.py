from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
import os
import requests
from neo4j import GraphDatabase

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "openai_api_key"


# Wait 60 seconds before connecting using these details, or login to https://console.neo4j.io to validate the Aura Instance is available
# Neo4j connection details
NEO4J_URL = "neo4j url"
NEO4J_USERNAME = "neo4j username"
NEO4J_PASSWORD = "neo4j password"
AURA_INSTANCEID= "aura_instance_id"
AURA_INSTANCENAME= "aura_instance_name"

# Initialize Neo4j graph and vector store
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Initialize the vector store with the Neo4j graph and OpenAI embeddings
# Note: The text_node_properties are the properties of the nodes in the Neo4j graph that will be used for similarity search
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='tasks',
    node_label="Task",
    text_node_properties=['name', 'description', 'status'],
    embedding_node_property='embedding',
)

# Initialize the Cypher chain with the Neo4j graph and vector store
# Note: The cypher_query_template is a Cypher query that retrieves courses based on qualification and interest
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-4'),
    qa_llm=ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_query_template=(
        "MATCH (q:Qualification {name: $qualification})-[:RELATES_TO]->(i:Interest {name: $interest})"
        "<-[:HAS_INTEREST {score: $interest_score}]-(u:User), (i)-[:LEADS_TO]->(c:Course) "
        "RETURN c.name AS course_name, c.description AS course_description, c.skill_level AS skill_level"
    )
)
# Initialize tools for the agent
# The tools are functions that the agent can use to perform specific tasks, such as similarity searches or executing Cypher queries.
# The tools are defined with a name, function, and description.
# The name is used to identify the tool, the function is the actual code that will be executed, and the description provides context for the tool's purpose.
# The tools are used by the agent to process user queries and provide responses.
# The tools are defined with a name, function, and description.
tools = [
    Tool(
        name="Career_Guidance",  # Use underscores instead of spaces
        func=vector_index.similarity_search,
        description="""Useful when you need to answer questions about career paths, 
        course details, or general guidance related to career choices.
        Use full question as input.
        """,
    ),
    Tool(
        name="Knowledge_Graph",  # Use underscores instead of spaces
        func=cypher_chain.run,
        description="""Useful when you need to answer questions about relationships 
        between qualifications, interests, and recommended courses. 
        Also useful for ranking and aggregating career recommendations.
        Use full question as input.
        """,
    ),
]

# Initialize the agent
# The agent is responsible for processing user queries and providing responses.
# It uses the tools defined above to perform specific tasks, such as similarity searches or executing Cypher queries.
# The agent is initialized with the tools, a language model (ChatOpenAI), and an agent type (OPENAI_FUNCTIONS).
# The agent type determines how the agent will interact with the tools and the language model.
mrkl = initialize_agent(
    tools,
    ChatOpenAI(temperature=0, model_name='gpt-4'),
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Function to get course recommendations based on qualification, interest, and interest assessment score
# The function takes the user's qualification, interest, and interest assessment score as input parameters.
# It constructs a query based on these inputs and uses the agent to get the response.
def get_course_recommendation(qualification: str, interest: str, interest_score: int) -> str:
    """
    Get course recommendations based on qualification, interest, and interest assessment score.

    Args:
        qualification (str): The user's qualification (e.g., "PCMC").
        interest (str): The user's interest (e.g., "Coding").
        interest_score (int): The user's interest assessment score (out of 5).

    Returns:
        str: A string containing the recommended courses.
    """
    # Construct the query based on inputs
    query = (
        f"Which courses are recommended for someone with {qualification} qualification, "
        f"an interest in {interest}, and an interest assessment score of {interest_score} out of 5?"
    )

    # Use the agent to get the response
    response = mrkl.run(query)
    
    # If the response is empty or indicates no results, return a default message
    if "I don't know" in response or "unable to provide" in response:
        return "No specific courses found. Consider exploring general courses related to your interest."
    
    return response


# Example usage
qualification = "PCMB"
interest = "Medicine"
interest_score = 3

recommendation = get_course_recommendation(qualification, interest, interest_score)
print(recommendation)


#graph databases are sql or no sql?
# Graph databases are a type of NoSQL database.

# 1. How neo4j is used here or what is the role of neo4j here?  
# Neo4j is used as a graph database to store and query information about qualifications, interests, and courses. 
# It allows for complex relationships between entities to be represented and queried efficiently.

# 2. Does the response here i am getting have any relation with the data in neo4j?
# Yes, the response is generated based on the data stored in Neo4j. 
# The queries executed against the Neo4j graph database retrieve relevant courses based on the user's qualification and interest.

# 3. what is the role of vector_index here?
# The vector_index is used to perform similarity searches in the Neo4j graph.
# It allows the system to find related courses or recommendations based on the user's input and the embeddings of the data stored in Neo4j.

# 4. What is embeddings and similarity search?
# Embeddings are numerical representations of data (like text) in a high-dimensional space.
# Similarity search is the process of finding data points that are close to a given point in this high-dimensional space,
# often using distance metrics like cosine similarity or Euclidean distance.

#what is the use of simialrity search in this code?
# The similarity search in this code is used to find relevant courses or recommendations based on the user's input and the embeddings of the data stored in Neo4j.
# It helps in identifying similar items or relationships between different pieces of data.

# 4. is the data points that similarity search finding is embeddings?
# Yes, the data points found through similarity search are typically embeddings.

#5. what is the purpose of finding close data point from high dimensional space?
# The purpose of finding close data points in high-dimensional space is to identify similar items or relationships between different pieces of data.

#actually when we do cypher query, similarity search is happening in the background or not?
# Yes, when a Cypher query is executed, similarity search may occur in the background to find relevant data points based on the query parameters.
#6. what is the role of embeddings in this code?
# The role of embeddings in this code is to provide a numerical representation of the data (qualifications, interests, courses) that captures their semantic meaning and relationships.

#what are the use of embeddings?
# Embeddings are used to represent data in a way that captures semantic meaning and relationships between different pieces of data.
# They are commonly used in natural language processing, recommendation systems, and machine learning tasks to improve the performance of models by providing a more meaningful representation of the data.

# 5. What is the role of cypher_chain here?
# The cypher_chain is responsible for executing Cypher queries against the Neo4j graph database.
# It retrieves information about qualifications, interests, and courses based on user input.

# 6. What is the role of mrkl here?
# mrkl is an agent that combines the tools (vector_index and cypher_chain) to process user queries and provide responses.

# 6. What is the role of tools here?
# The tools are functions that the agent can use to perform specific tasks, such as similarity searches or executing Cypher queries.

# 7. What is the role of ChatOpenAI here?
# ChatOpenAI is used as the language model for generating responses and executing Cypher queries.

# 8. What is the role of GraphCypherQAChain here?
# GraphCypherQAChain is responsible for handling the interaction between the language model and the Neo4j graph database.
# It executes Cypher queries and retrieves relevant information based on user input.

# 10. What is the role of Neo4jVector here?
# Neo4jVector is used to perform similarity searches in the Neo4j graph database.

# 11. What is the role of Neo4jGraph here?
# Neo4jGraph is used to represent the Neo4j graph database and allows for querying and interacting with the graph.

# 12. What is the role of GraphDatabase here?
# GraphDatabase is used to establish a connection to the Neo4j database and execute queries.
