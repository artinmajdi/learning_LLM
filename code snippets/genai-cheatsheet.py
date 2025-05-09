"""
GEN AI DATA SCIENTIST INTERVIEW CHEATSHEET
For American Airlines IT Operations Research and Advanced Analytics position
"""

#######################
# RAG (Retrieval Augmented Generation)
#######################

# Imports
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Pinecone, Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
import pinecone

# Document Processing
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Embeddings
def get_embeddings(embedding_type="openai"):
    if embedding_type == "openai":
        return OpenAIEmbeddings(
            azure_deployment="embedding-deployment",
            openai_api_version="2023-05-15"
        )
    elif embedding_type == "huggingface":
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Stores
def create_vectorstore(chunks, embeddings, store_type="faiss"):
    if store_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)
    elif store_type == "pinecone":
        pinecone.init(api_key="your-api-key", environment="your-env")
        index_name = "your-index"
        return Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    elif store_type == "weaviate":
        return Weaviate.from_documents(chunks, embeddings, url="your-weaviate-url")

# LLM Setup
def get_llm(provider="azure"):
    if provider == "azure":
        return AzureOpenAI(
            deployment_name="gpt-4-turbo",
            model_name="gpt-4-turbo",
            openai_api_version="2023-05-15"
        )

# RAG Chain
def create_rag_chain(vectorstore, llm):
    prompt_template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

#######################
# PROMPT ENGINEERING
#######################

# System Prompt Template for IT Operations
system_prompt = """You are an AI assistant for American Airlines IT operations.
Follow these guidelines:
1. Only provide information found in the retrieved context
2. If information is not in the context, respond with "I don't have enough information"
3. Cite the source of information when possible
4. Do not make assumptions about IT systems or procedures
5. Format technical instructions clearly with step-by-step guidance

Context:
{context}

Question: {question}
"""

# Few-shot examples for IT troubleshooting
few_shot_examples = """
Example 1:
Question: How do I reset VPN access?
Answer: To reset VPN access, first open the Self-Service Portal at https://selfservice.aa.com, navigate to "Access Management", select "VPN Reset", and follow the verification steps. If you encounter errors, contact the IT Help Desk at extension 1234.

Example 2:
Question: What is the procedure for server maintenance?
Answer: Server maintenance requires following the change management process in ServiceNow. Create a Change Request (CR) at least 48 hours in advance, get approval from the IT Operations Manager, and schedule the maintenance during approved windows (usually Sundays 2-6 AM CST).
"""

# Structured Output Format
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

response_schemas = [
    ResponseSchema(name="answer", description="The answer to the query based only on provided context"),
    ResponseSchema(name="sources", description="References to the specific documents used"),
    ResponseSchema(name="confidence", description="How confident you are in this answer (high/medium/low)")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

#######################
# VECTOR DATABASES
#######################

# FAISS
import faiss
import numpy as np

# Create FAISS index
def create_faiss_index(embeddings, dimension=384):
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    # Or use IVF for larger datasets
    # index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100)
    
    # Convert to numpy and add to index
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

# Search FAISS index
def search_faiss(index, query_embedding, k=5):
    query_np = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_np, k)
    return distances[0], indices[0]

# Pinecone
import pinecone

def setup_pinecone(api_key, environment):
    pinecone.init(api_key=api_key, environment=environment)
    
def create_pinecone_index(index_name, dimension=384):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
    return pinecone.Index(index_name)

def upsert_to_pinecone(index, vectors, ids, metadata=None):
    if metadata is None:
        metadata = [{} for _ in ids]
    
    vectors_with_ids = list(zip(ids, vectors, metadata))
    index.upsert(vectors=vectors_with_ids)

def query_pinecone(index, query_vector, top_k=5):
    results = index.query(query_vector, top_k=top_k, include_metadata=True)
    return results

# Weaviate
import weaviate
from weaviate.auth import AuthApiKey

def setup_weaviate(url, api_key=None):
    auth_client = None
    if api_key:
        auth_client = AuthApiKey(api_key=api_key)
    
    client = weaviate.Client(url=url, auth_client=auth_client)
    return client

def create_weaviate_schema(client, class_name="Document"):
    schema = {
        "classes": [{
            "class": class_name,
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["object"]}
            ]
        }]
    }
    client.schema.create(schema)

def hybrid_search_weaviate(client, query, class_name="Document", alpha=0.5, limit=5):
    result = (
        client.query
        .get(class_name, ["content", "metadata"])
        .with_hybrid(
            query=query,
            alpha=alpha
        )
        .with_limit(limit)
        .do()
    )
    return result["data"]["Get"][class_name]

#######################
# AZURE AI FOUNDRY
#######################

from azure.ai.resources import AIClient
from azure.identity import DefaultAzureCredential

# Set up AI client
def setup_ai_client(subscription_id, resource_group, workspace_name):
    credential = DefaultAzureCredential()
    ai_client = AIClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    return ai_client

# Deploy model on Azure AI Foundry
def deploy_model(ai_client, model_name, deployment_name, instance_type, instance_count=1):
    deployment_config = {
        "model_name": model_name,
        "instance_type": instance_type,
        "instance_count": instance_count,
        "environment_variables": {
            "TRANSFORMERS_CACHE": "/tmp/huggingface"
        },
        "scaling_settings": {
            "min_instances": 1,
            "max_instances": 5
        }
    }
    
    poller = ai_client.deployments.begin_create_or_update(
        name=deployment_name,
        deployment=deployment_config
    )
    
    return poller

# Monitor model deployment
from azure.ai.ml import MLClient

def monitor_deployment(ml_client, deployment_name, endpoint_name):
    deployment = ml_client.online_deployments.get(
        name=deployment_name,
        endpoint_name=endpoint_name
    )
    
    metrics = ml_client.online_deployments.get_metrics(
        name=deployment_name,
        endpoint_name=endpoint_name,
        start_time="2025-04-30T00:00:00Z",
        end_time="2025-05-07T00:00:00Z"
    )
    
    return {
        "status": deployment.provisioning_state,
        "metrics": metrics
    }

#######################
# MODEL EVALUATION & EXPLAINABILITY
#######################

# LLM Evaluation with RAGAS
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.metrics.critique import harmfulness

def evaluate_llm_responses(queries, responses, ground_truths):
    scores = {
        "faithfulness": [],
        "relevancy": [],
        "precision": [],
        "harmfulness": []
    }
    
    for query, response, ground_truth in zip(queries, responses, ground_truths):
        scores["faithfulness"].append(faithfulness.score(response, ground_truth))
        scores["relevancy"].append(answer_relevancy.score(response, query))
        scores["precision"].append(context_precision.score(response, query))
        scores["harmfulness"].append(harmfulness.score(response))
    
    return {k: sum(v)/len(v) for k, v in scores.items()}

# SHAP for ML Explainability
import shap
import matplotlib.pyplot as plt

def explain_model(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    # Global feature importance
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    # Sample explanation plot
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("sample_explanation.png")
    
    return shap_values

#######################
# MLOPS & MODEL MONITORING
#######################

import mlflow
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Setup MLflow
def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name)
    return mlflow

# Log model with MLflow
def log_model_mlflow(model, model_name, metrics, params, model_signature=None):
    with mlflow.start_run():
        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            model_name,
            signature=model_signature
        )
        
        run_id = mlflow.active_run().info.run_id
    
    return run_id

# Model drift detection
def detect_model_drift(reference_data, current_data, features, threshold=0.05):
    """Simple drift detection based on statistical tests"""
    from scipy import stats
    
    drift_metrics = {}
    
    for feature in features:
        # Use KS test for numerical features
        ks_stat, p_value = stats.ks_2samp(
            reference_data[feature],
            current_data[feature]
        )
        
        drift_metrics[feature] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
    
    # Check overall drift
    drift_detected = any(m['drift_detected'] for m in drift_metrics.values())
    
    return {
        'drift_detected': drift_detected,
        'feature_metrics': drift_metrics
    }

#######################
# LLM FINE-TUNING
#######################

from datasets import Dataset
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    TrainingArguments,
    Trainer, 
    DataCollatorForLanguageModeling
)

# Prepare dataset for fine-tuning
def prepare_llm_dataset(data, instruction_template="<s>[INST] {instruction} [/INST] {response} </s>"):
    # Format data
    formatted_data = []
    for item in data:
        instruction = item["instruction"]
        response = item["response"]
        formatted_text = instruction_template.format(
            instruction=instruction,
            response=response
        )
        formatted_data.append({"text": formatted_text})
    
    # Create HF dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    return dataset

# Setup tokenizer for fine-tuning
def tokenize_dataset(dataset, tokenizer, max_length=512):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Fine-tune LLM
def finetune_llm(model_id, tokenized_dataset, output_dir, epochs=3, batch_size=4):
    # Load model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id, 
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=1000,
        bf16=True,  # Use BF16 if available
        learning_rate=2e-5,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    return model, tokenizer

#######################
# IT OPERATIONS ANALYTICS
#######################

# IT Incident Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_it_incidents(incidents_df, n_clusters=10):
    # Extract descriptions
    descriptions = incidents_df['description'].fillna('')
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        stop_words='english',
        min_df=5,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(descriptions)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    incidents_df['cluster'] = kmeans.fit_predict(X)
    
    # Get top terms per cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_centers = kmeans.cluster_centers_
    
    top_terms = {}
    for i in range(n_clusters):
        center = cluster_centers[i]
        top_indices = center.argsort()[-10:][::-1]  # Top 10 terms
        top_terms[i] = [feature_names[idx] for idx in top_indices]
    
    # Summarize clusters
    cluster_summary = incidents_df.groupby('cluster').agg({
        'incident_id': 'count',
        'priority': lambda x: x.value_counts().index[0]  # Most common priority
    }).reset_index()
    
    cluster_summary.columns = ['cluster', 'incident_count', 'most_common_priority']
    cluster_summary['top_terms'] = cluster_summary['cluster'].map(top_terms)
    
    return cluster_summary

# Predictive Maintenance for IT Systems
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_predictive_maintenance_model(system_logs_df, features, target='failure_within_24h'):
    # Feature engineering
    X = system_logs_df[features]
    y = system_logs_df[target]
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ))
    ])
    
    # Train model
    pipeline.fit(X, y)
    
    # Feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return pipeline, importance_df
