## The Langchain integrations related to Amazon AWS platform : Amazon Bedrock 

Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs)from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a signle API, along  with a broad set of capabilities you need to build generative AI applications with security, privacy and responsible AI. Using Amazon Bedrock, you can easly experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Agmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serveless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with.

# LLMs 

```python

from lagchain_community.llms import Bedrock

llm = Bedrock(credentials_profile_name="bedrock-admin", model_id="amazon.titan-text-express-v1")

```

### Custom models 

```python

custom_llm= Bedrock(
    credentials_profile_name="bedrock-admin",
    provider="cohere"
    model_id="<Custom model ARN>", # ARN like 'arn:aws:bedrock:...' obtained via provisioning the custom model
    model_kwargs={"temperature": 1},
    streaming=True, 
    callbacks =[StreamingStdOutCallbackHandler]
)

```

### Using the LLM in a conversation chain

```python

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(llm = llm, verbose=True, memory= ConversationBufferMemory())

conversation.predict(input= "Hi there!")

```

# Embedding Models 


```python

from langchain_community.embeddings import BedrockEmbeddings 

embeddings = BedrocEmbeddings( credentials_profile_name="bedrock-admin", region_name="us-east-1")

```

```python

embeddings.embed_query("This is a content of the document")

```

```python

embeddings.embed_documents(["This is a content of the document", "This is another document"])

```

# Amazon Bedrock (Knowledge Bases)

Knowledge bases for Amazon Bedrock is an Amazon Web Services (AWS) offering which lets you quickly build RAG applications by using your private data to customize foundation model response

Implementing RAG requires organization to perfom several cumbersome steps to convert data into embeddings(vectors), store the embeddings in a specialized vector database, and build custom integrations into the database and retrieve text relevant to the user's query. This can be time-consuming and inefficient.

With Knowledge Bases for Amazon Bedrock, simply point to the location of your data in Amazon S3, and knowledge bases for Amazon Bedrock takes care of the entire ingestion workflow into your vector database. If you do not have an existing vector database, Amazon Bedrock creates an Amazon OpenSearch Serverless vector store for you. For retrievals, use the Langchain-Amazon Bedrock integration via the Retrieve API to retrieve relevent results for a user query from knowledge bases.

## The Knowledge Bases Retriever

```python

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="PUIJP4EQUA"
    retrieval_config={"vectorSearchConfiguration" : {"numberOfResults": 4}},
)

```

## Using the Knowledge Bases Retriever in a QA chain 

- RestrievalQA : Chain of question-answering against an index

```python

from botocore.client import Config 
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock

model_kwargs_claude = {"temperature": 0, "top_k": 10, "max_tokens_to_sample":3000}

llm = Bedrock(model_id="anthropic.claude-v2", model_kwargs=model_kwargs_claude)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever= retriever, return_source_documents=True)

qa_chain(query)

```

# Tools : AWS Lambda

Amazon AWS Lambda is a serverless computing service provided by AWS. It helps developpers to build and run applications and services without provisioning or managing servers. This serverless architecture enables you to focus on writing and deploying code, while AWS automatically takes care of scaling, patching, and managing the infrastructure required to run your applications.


By including the AWS Lambda, in the list of tools provided to an Agent, you can grant your Agent the ability to invoke code running in your AWS Cloud for whatever purposes you need.

When an Agent uses the AWS Lambda tool, it will provide an argument of type string which will in turn be passed into the Lambda function via the event parameter.

In order for an Agent to use the tool, you must provide it with the name and description that match the functionality of your lambda function's logic.

You must also provide the name of your function.

```python 

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature = 0 )

tools = load_tools(
    ["awslambda"],
    awslambda_tool_name = "email-sender"
    awslambda_tool_description = "Sends an email with the specified content to test@test.com "
    function_name = "testFuncion1",
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("Send an email to test@test.com saying Hello world.")

```
