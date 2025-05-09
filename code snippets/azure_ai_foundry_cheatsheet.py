# Azure AI Foundry Cheatsheet for Gen AI Data Scientist Interview (American Airlines)

"""
This cheatsheet provides concise answers and conceptual code snippets 
for potential Azure AI Foundry questions relevant to American Airlines.
"""

# === 1. Understanding Azure AI Foundry ===

# Q: What is Azure AI Foundry, and how can it benefit American Airlines?
"""
Answer:
Azure AI Foundry is a platform that simplifies building, deploying, and managing 
enterprise-grade AI applications, particularly Generative AI. 
It integrates various Azure AI services (Azure ML, Azure OpenAI, Azure AI Search, etc.) 
into a unified experience.

Benefits for American Airlines (AA):
- Faster development of Gen AI solutions (e.g., customer service bots, personalized travel offers).
- Scalable and secure deployment of models.
- Access to pre-built models (e.g., from Meta, Mistral) and foundation models (OpenAI's GPT series).
- Tools for Responsible AI, prompt engineering (Prompt Flow), and MLOps.
- Improved operational efficiency and enhanced customer experiences.
"""

# === 2. Core Components of Azure AI Foundry ===

# Q: What are some key components of Azure AI Foundry?
"""
Answer:
- Azure Machine Learning (AzureML): Core platform for MLOps, model training, deployment, and management.
- Azure OpenAI Service: Provides access to powerful foundation models like GPT-4, GPT-3.5-turbo, DALL-E, and embeddings, with enterprise security and responsible AI features.
- Azure AI Search (formerly Cognitive Search): For building RAG solutions by indexing and searching AA's private data (e.g., flight manuals, customer feedback, maintenance logs).
- Prompt Flow: A tool within AzureML for developing, evaluating, and deploying LLM prompts and chains (e.g., for RAG applications).
- Model Catalog: A hub in AzureML to discover, evaluate, and use foundation models and open-source models.
- Azure AI Studio: A unified portal that brings these services together for a streamlined development experience.
"""

# === 3. Prompt Engineering & Prompt Flow ===

# Q: How would you use Prompt Flow in Azure AI Foundry to develop a Gen AI application for AA?
"""
Answer:
Prompt Flow helps create executable flows that link LLMs, prompts, Python tools, and other resources. 
For AA, I could use it to build a customer service assistant:

1.  **Design the Flow:** 
    - Input: Customer query (e.g., "What's the baggage allowance for my flight to LHR?").
    - Steps:
        - Process input (e.g., extract intent, entities).
        - Retrieve relevant AA policy documents using Azure AI Search (RAG pattern).
        - Construct a prompt with the customer query and retrieved context.
        - Call an Azure OpenAI model (e.g., GPT-4) with the prompt.
        - Format the output for the customer.
2.  **Iterate and Test:** Use Prompt Flow's visual editor to build and test each node. 
    Evaluate prompt variations and LLM responses for accuracy and helpfulness.
3.  **Evaluate:** Run batch tests with diverse queries and compare outputs against ground truth or quality metrics.
4.  **Deploy:** Deploy the flow as a managed online endpoint in AzureML for real-time inference.
"""

# Conceptual Prompt Flow Snippet (Illustrative Python, not direct Prompt Flow SDK usage)
# This shows the logic you'd build visually in Prompt Flow

def customer_service_flow(customer_query: str, vector_store_retriever):
    # from promptflow.core import tool (this would be implicit in Prompt Flow)

    # @tool (decorator for a Prompt Flow node)
    def retrieve_aa_documents(query: str, retriever):
        # Simulates retrieving docs from Azure AI Search integrated vector store
        print(f"Retrieving documents for: {query}")
        # relevant_docs = retriever.get_relevant_documents(query)
        # For AA: flight policies, baggage rules, loyalty program details
        # Example: "Baggage allowance for flight AA123 to LHR on 2024-12-25"
        # Would fetch documents related to 'baggage allowance', 'international flights', 'economy class' etc.
        if "baggage allowance" in query.lower():
            return ["Document: Economy class international flights allow 1 checked bag up to 23kg."]
        return ["Document: General flight information."]

    # @tool
    def generate_response(query: str, retrieved_docs: list):
        # Simulates calling Azure OpenAI model via Prompt Flow LLM tool
        print(f"Generating response for: {query} with context: {retrieved_docs}")
        # prompt = f"Answer the question: '{query}' using only the following context from American Airlines: {retrieved_docs}. If the context doesn't have the answer, say you don't know."
        # response = AzureOpenAI_LLM_call(prompt)
        if "baggage allowance" in query.lower() and retrieved_docs:
            return f"Based on AA policy: {retrieved_docs[0].split(':')[1].strip()}"
        return "I can help with general flight information. For specific baggage queries, please provide more details or check our website."

    # Flow execution
    context_docs = retrieve_aa_documents(customer_query, vector_store_retriever)
    final_answer = generate_response(customer_query, context_docs)
    return final_answer

# print(f"Example Flow Output: {customer_service_flow('What is the baggage allowance for international flights?', None)}")

# === 4. Retrieval Augmented Generation (RAG) ===

# Q: Explain how you would implement a RAG solution for American Airlines using Azure AI Foundry.
"""
Answer:
1.  **Data Ingestion & Preparation:** 
    - Identify AA's internal data sources (e.g., maintenance manuals, HR policies, customer feedback databases, flight operations procedures).
    - Ingest and chunk these documents into manageable sizes.
    - Store them in Azure Blob Storage.
2.  **Vectorization & Indexing:**
    - Use an embedding model from Azure OpenAI (e.g., text-embedding-ada-002) to convert document chunks into vector embeddings.
    - Store these embeddings in Azure AI Search, which acts as a vector database. This creates a searchable index of AA's knowledge.
3.  **Retrieval:**
    - When a user query comes in (e.g., "What is the pre-flight check procedure for a Boeing 737?"), 
      embed the query using the same embedding model.
    - Perform a vector similarity search in Azure AI Search to find the most relevant document chunks (context).
4.  **Augmentation & Generation:**
    - Construct a prompt that includes the original user query and the retrieved context chunks.
    - Send this augmented prompt to an Azure OpenAI model (e.g., GPT-4) to generate a concise, context-aware answer.
5.  **Integration & Orchestration:**
    - Use Prompt Flow to build, test, and deploy this RAG pipeline as an API.
"""

# Conceptual RAG Snippet (Python using LangChain with Azure services)
"""
# from langchain_community.document_loaders import PyPDFLoader # For AA's PDF manuals
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain_community.vectorstores import AzureSearch
# from langchain.chains import RetrievalQA
# import os

# # Setup (assuming Azure resources are configured and API keys are in env vars)
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
# EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
# LLM_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
# AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
# AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
# INDEX_NAME = "american-airlines-knowledge-base"

# # 1. Load AA Documents (e.g., maintenance guides)
# # loader = PyPDFLoader("path/to/AA_maintenance_manual.pdf") 
# # documents = loader.load()
# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# # texts = text_splitter.split_documents(documents)

# # 2. Embeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
#     openai_api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     openai_api_version=AZURE_OPENAI_API_VERSION
# )

# # 3. Vector Store (Azure AI Search)
# # If creating for the first time:
# # vector_store = AzureSearch.from_documents(
# #     documents=texts, 
# #     embedding=embeddings, 
# #     azure_search_endpoint=f"https://{AZURE_AI_SEARCH_SERVICE_NAME}.search.windows.net",
# #     azure_search_key=AZURE_AI_SEARCH_API_KEY,
# #     index_name=INDEX_NAME
# # )
# # If loading an existing index:
# # vector_store = AzureSearch(
# #     azure_search_endpoint=f"https://{AZURE_AI_SEARCH_SERVICE_NAME}.search.windows.net",
# #     azure_search_key=AZURE_AI_SEARCH_API_KEY,
# #     index_name=INDEX_NAME,
# #     embedding_function=embeddings.embed_query
# # )

# # 4. LLM
# llm = AzureChatOpenAI(
#     azure_deployment=LLM_DEPLOYMENT_NAME,
#     openai_api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     temperature=0
# )

# # 5. RetrievalQA Chain
# # qa_chain = RetrievalQA.from_chain_type(
# #     llm=llm, 
# #     chain_type="stuff", 
# #     retriever=vector_store.as_retriever()
# # )

# # Example query for AA maintenance
# # query = "What is the standard procedure for hydraulic system checks on a Boeing 777?"
# # response = qa_chain.run(query)
# # print(response)
"""

# === 5. Fine-tuning vs. RAG ===

# Q: When would you recommend fine-tuning a model versus using RAG for an AA use case?
"""
Answer:
- **Use RAG when:**
    - The primary need is to ground the LLM in AA's specific, factual, and frequently updated data (e.g., flight schedules, internal policies, FAQs, maintenance procedures).
    - You need to reduce hallucinations and provide attributable answers by citing sources.
    - The domain knowledge is extensive and can be effectively retrieved.
    - Example for AA: A chatbot answering customer queries about baggage policies, flight status, or loyalty programs using AA's latest documents.

- **Consider Fine-tuning when:**
    - You need to adapt the LLM's style, tone, or format to match AA's branding or specific communication needs (e.g., generating marketing copy in AA's voice).
    - You need to teach the model a new skill or improve its performance on a very specific task where RAG alone is insufficient due to nuanced understanding requirements.
    - You have a high-quality, curated dataset of prompt-completion pairs for the specific task.
    - Example for AA: Fine-tuning a model to summarize pilot reports into a standardized format, or to generate empathetic responses for customer complaints in a specific AA tone.

- **Hybrid Approach:** Often, a combination is best. RAG provides the knowledge, and a fine-tuned model might better understand how to use that knowledge or interact in the desired style.
"""

# === 6. Model Deployment & MLOps ===

# Q: How does Azure AI Foundry support MLOps for Gen AI models at American Airlines?
"""
Answer:
Azure AI Foundry, through Azure Machine Learning, provides robust MLOps capabilities:
- **Model Management:** Register and version LLMs, fine-tuned models, and prompt flows in the AzureML Model Registry.
- **Endpoint Deployment:** Deploy models and prompt flows as scalable and secure managed online endpoints (for real-time inference) or batch endpoints.
    - For AA: Deploying a customer service bot, a fare prediction model, or a document summarization tool.
- **Monitoring:** Monitor model performance, operational metrics (latency, errors), and data drift. Set up alerts for issues.
    - For AA: Tracking the accuracy of a baggage claim prediction model, or the helpfulness scores of a customer support bot.
- **CI/CD Integration:** Use Azure DevOps or GitHub Actions to automate the build, test, and deployment pipelines for Gen AI applications.
- **Prompt Flow & Connections:** Manage connections to Azure OpenAI, Azure AI Search, and other services securely within AzureML workspaces.
- **Responsible AI Dashboard:** Track fairness, explainability, and other responsible AI metrics.
"""

# Conceptual MLOps Snippet (Azure CLI for deploying a model/prompt flow)
"""
# # Ensure you are logged in: az login
# # Set subscription: az account set --subscription <YOUR_SUBSCRIPTION_ID>

# # Variables
# RESOURCE_GROUP="AA-GenAI-RG"
# AML_WORKSPACE="AA-AML-Workspace"
# ENDPOINT_NAME="aa-customer-service-bot-endpoint"
# # Assume a prompt flow or model is registered in AzureML
# FLOW_OR_MODEL_NAME="aa_customer_service_flow"
# FLOW_OR_MODEL_VERSION=1 # or specific version

# # Create an endpoint (if it doesn't exist)
# # az ml online-endpoint create --name $ENDPOINT_NAME -f endpoint.yml --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE

# # Example endpoint.yml (simplified for a prompt flow):
# # $schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
# # name: aa-customer-service-bot-endpoint
# # auth_mode: key
# # traffic:
# #   blue: 100 # 100% traffic to the blue deployment

# # Create a deployment for the flow/model under the endpoint
# # az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f deployment.yml --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE --all-traffic

# # Example deployment.yml (simplified for a prompt flow deployment):
# # $schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
# # name: blue
# # endpoint_name: aa-customer-service-bot-endpoint
# # model: azureml_promptflow_flow_runtime # For prompt flow this might be different, often references the flow folder or a registered flow
# #   # path: ../path_to_your_flow_folder # if deploying a local flow folder
# #   # name: $FLOW_OR_MODEL_NAME # if deploying a registered flow or model
# #   # version: $FLOW_OR_MODEL_VERSION
# # instance_type: Standard_DS3_v2
# # instance_count: 1
# # # For a registered flow, it might look like:
# # # flow: $FLOW_OR_MODEL_NAME # Name of the registered flow
# # # version: $FLOW_OR_MODEL_VERSION

# # Invoke the endpoint
# # az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file sample-request.json --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE

# # Example sample-request.json (for a flow expecting 'customer_query'):
# # {
# #   "inputs": {
# #     "customer_query": "What is my baggage allowance?"
# #   }
# # }
"""

# === 7. Responsible AI ===

# Q: How would you ensure Responsible AI practices when developing a Gen AI solution for AA using Azure AI Foundry?
"""
Answer:
Azure AI Foundry provides tools and frameworks to implement Responsible AI principles:
1.  **Safety & Security:**
    -   Use Azure OpenAI's content filtering capabilities to mitigate harmful content generation (hate speech, self-harm, violence, sexual content).
    -   Implement input validation and output moderation for AA's applications (e.g., a customer-facing chatbot).
    -   Leverage Azure security features for data protection and access control.
2.  **Fairness & Inclusivity:**
    -   Evaluate models for potential biases, especially for applications affecting diverse customer groups (e.g., personalized offers, crew scheduling tools).
    -   Use AzureML's fairness assessment tools to identify and mitigate biases.
    -   Ensure training data for fine-tuning is representative of AA's diverse customer base and employee population.
3.  **Transparency & Explainability:**
    -   For RAG systems, provide citations to the source documents used for generating answers, increasing trust and allowing verification.
    -   While LLMs are complex, use SHAP or similar techniques for feature importance if a smaller, interpretable model is part of the Gen AI pipeline.
    -   Clearly communicate the capabilities and limitations of AI systems to AA users and customers.
4.  **Accountability & Governance:**
    -   Use AzureML's model lineage and audit trails.
    -   Establish clear ownership and review processes for AI applications within AA.
    -   Regularly review and update models and safety systems.
5.  **Reliability & Robustness:**
    -   Thoroughly test applications under various conditions, including edge cases.
    -   Monitor performance and implement fallback mechanisms.
    -   For AA: Ensure a flight delay prediction model is robust to unexpected events.
"""

# === 8. Use Cases for American Airlines ===

# Q: Suggest some specific Gen AI use cases for American Airlines that can be built using Azure AI Foundry.
"""
Answer:
1.  **Enhanced Customer Service Chatbot:** 
    -   Handles complex queries (flight changes, cancellations, loyalty program, baggage info) by using RAG on AA's policies and real-time flight data.
    -   Personalized responses based on customer history.
    -   Seamless handover to human agents if needed.
2.  **Internal Knowledge Assistant for Employees:**
    -   Pilots: Quick access to flight manuals, emergency procedures, weather updates.
    -   Maintenance Crew: Troubleshooting guides, part information, repair histories (RAG on technical docs).
    -   Gate Agents: Instant answers to policy questions, passenger rebooking assistance.
3.  **Personalized Travel Offers and Ancillary Revenue:**
    -   Generate personalized travel package recommendations, seat upgrade offers, or lounge access suggestions based on customer preferences and travel patterns.
4.  **Automated Summarization of Reports:**
    -   Summarize customer feedback, incident reports, or daily operational logs for quick review by management.
5.  **Drafting Communications:**
    -   Assist in drafting internal memos, customer notifications for delays, or marketing copy in AA's brand voice (potentially with fine-tuning).
6.  **Synthetic Data Generation for Training:**
    -   Generate realistic synthetic data for training other ML models where real data is scarce or sensitive (e.g., rare incident scenarios for crew training simulations).
7.  **Code Generation & Modernization Assistant:**
    -   Assist developers in modernizing legacy systems or generating boilerplate code for new applications related to AA's operations.
"""

print("Azure AI Foundry Cheatsheet for American Airlines - Gen AI Data Scientist role.")
print("This file contains Q&A and conceptual code snippets.")
