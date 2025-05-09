"""
GEN AI DATA SCIENTIST CHEATSHEET
American Airlines Interview - Code Snippets Reference
"""

# =====================================================================
# 1. RAG PIPELINE IMPLEMENTATIONS
# =====================================================================

# --- LangChain RAG with OpenAI ---
def langchain_openai_rag():
    """Basic RAG with LangChain + OpenAI"""
    import os
    from dotenv import load_dotenv
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA

    # Load documents
    loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create retriever and LLM
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Other options: "map_reduce", "refine"
        retriever=retriever,
        return_source_documents=True
    )
    
    # Query
    result = qa_chain({"query": "What is the baggage policy?"})
    return result["result"]  # Access source_documents with result["source_documents"]

# --- LangChain RAG with Google Gemini ---
def langchain_gemini_rag():
    """RAG with LangChain + Google Gemini"""
    import os
    from dotenv import load_dotenv
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    
    # Create embeddings and LLM
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Rest of the pipeline is the same as OpenAI version
    # ...

# --- LlamaIndex RAG ---
def llamaindex_rag():
    """RAG with LlamaIndex"""
    import os
    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms import OpenAI
    from llama_index.embeddings import OpenAIEmbedding
    
    # Load documents
    documents = SimpleDirectoryReader("./data").load_data()
    
    # Create LLM and embedding model
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    
    # Create index and query engine
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    
    # Query
    response = query_engine.query("What is American Airlines' policy on flight changes?")
    return response.response

# --- Azure OpenAI + Azure AI Search RAG ---
def azure_openai_search_rag():
    """RAG with Azure OpenAI and Azure AI Search"""
    import os
    from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
    from langchain_community.vectorstores import AzureSearch
    from langchain.chains import RetrievalQA
    
    # Azure OpenAI setup
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    llm = AzureChatOpenAI(
        azure_deployment="gpt-35-turbo",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Azure AI Search setup
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name="aa-documents",
        embedding_function=embeddings.embed_query
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

# =====================================================================
# 2. EXPLAINABLE AI - SHAP & FEATURE IMPORTANCE
# =====================================================================

def xgboost_shap_example():
    """XGBoost with SHAP explanations"""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    # Train XGBoost model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Built-in feature importance
    built_in_importance = model.get_booster().get_score(importance_type='gain')
    
    # SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # For binary classification
    if len(shap_values.values.shape) == 3:  # (samples, features, classes)
        shap_values_for_class_1 = shap_values.values[:, :, 1]  # For positive class
        expected_value_class_1 = explainer.expected_value[1]
    else:  # (samples, features)
        shap_values_for_class_1 = shap_values.values
        expected_value_class_1 = explainer.expected_value
    
    # Global importance (bar plot)
    shap.summary_plot(shap_values_for_class_1, X_test, plot_type="bar")
    
    # Feature distribution impact (beeswarm)
    shap.summary_plot(shap_values_for_class_1, X_test)
    
    # Local explanation (waterfall plot for one prediction)
    passenger_index = 0
    individual_shap_values = shap.Explanation(
        values=shap_values_for_class_1[passenger_index,:],
        base_values=expected_value_class_1,
        data=X_test.iloc[passenger_index,:],
        feature_names=X_test.columns
    )
    shap.plots.waterfall(individual_shap_values)
    
    # Feature interaction
    shap.dependence_plot(
        "FlightDelayMinutes",  # Feature to analyze
        shap_values_for_class_1,
        X_test,
        interaction_index="IsEliteStatus"  # Interaction feature
    )

# =====================================================================
# 3. FINE-TUNING LLMs
# =====================================================================

# --- OpenAI Fine-tuning ---
def openai_finetuning():
    """OpenAI fine-tuning example"""
    import openai
    import json
    import os
    
    # Prepare training data
    training_data = [
        {"messages": [
            {"role": "system", "content": "You are an American Airlines customer service assistant."},
            {"role": "user", "content": "I need to change my flight."},
            {"role": "assistant", "content": "I'd be happy to help you change your flight. Could you please provide your booking reference and the new flight details you're interested in?"}
        ]},
        # More examples...
    ]
    
    # Save to JSONL file
    with open("training_data.jsonl", "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")
    
    # Create fine-tuning job
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Upload file
    file = client.files.create(
        file=open("training_data.jsonl", "rb"),
        purpose="fine-tune"
    )
    
    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-3.5-turbo"
    )
    
    # Use fine-tuned model
    completion = client.chat.completions.create(
        model=f"ft:{job.fine_tuned_model}",  # Use the fine-tuned model ID
        messages=[
            {"role": "system", "content": "You are an American Airlines customer service assistant."},
            {"role": "user", "content": "I missed my connecting flight."}
        ]
    )

# --- Azure OpenAI Fine-tuning ---
def azure_openai_finetuning():
    """Azure OpenAI fine-tuning"""
    from azure.ai.openai import OpenAIClient
    from azure.ai.openai.models import FineTuningJob, FineTuningJobData
    from azure.identity import DefaultAzureCredential
    
    # Create Azure OpenAI client
    client = OpenAIClient(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        credential=DefaultAzureCredential()
    )
    
    # Create fine-tuning job
    job = client.fine_tuning.create(
        model="gpt-35-turbo",
        training_file="file-id",  # File ID from upload
        hyperparameters={
            "n_epochs": 3
        }
    )

# =====================================================================
# 4. PROMPT ENGINEERING & LANGCHAIN
# =====================================================================

def langchain_prompt_techniques():
    """LangChain prompt engineering examples"""
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain.chains import LLMChain
    from langchain_openai import ChatOpenAI
    
    # Basic prompt template
    prompt = PromptTemplate(
        input_variables=["flight_number", "issue"],
        template="You are an American Airlines customer service agent. A customer with flight {flight_number} has the following issue: {issue}. How would you respond?"
    )
    
    # Few-shot learning prompt
    examples = [
        {"query": "Flight delayed", "response": "I understand how frustrating delays can be..."},
        {"query": "Baggage lost", "response": "I'm sorry to hear about your lost baggage..."}
    ]
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=PromptTemplate(input_variables=["query", "response"], template="{query}: {response}"),
        prefix="Answer customer queries in a helpful, empathetic tone:\n\n",
        suffix="\nQuery: {input}\nResponse:",
        input_variables=["input"]
    )
    
    # Chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an American Airlines customer service agent."),
        ("human", "{query}")
    ])
    
    # Structured output parsing
    class FlightRecommendation(BaseModel):
        flight_number: str = Field(description="The recommended flight number")
        departure_time: str = Field(description="Departure time in local timezone")
        arrival_time: str = Field(description="Arrival time in local timezone")
        price: float = Field(description="Price in USD")
    
    parser = JsonOutputParser(pydantic_object=FlightRecommendation)
    
    structured_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an American Airlines flight recommendation system. Return a JSON object with flight details."),
        ("human", "Find me a flight from {origin} to {destination} on {date}."),
        ("system", f"Format instructions: {parser.get_format_instructions()}")
    ])
    
    # Chain components together
    llm = ChatOpenAI()
    chain = structured_prompt | llm | parser
    
    result = chain.invoke({"origin": "DFW", "destination": "JFK", "date": "2025-06-01"})

# =====================================================================
# 5. EVALUATION METRICS FOR GEN AI
# =====================================================================

def genai_evaluation():
    """Evaluation metrics for Gen AI models"""
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from ragas.metrics.critique import harmfulness
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # RAGAS for RAG evaluation
    eval_dataset = Dataset.from_dict({
        "question": ["What is AA's baggage policy?"],
        "answer": ["American Airlines allows one carry-on bag and one personal item for free..."],
        "contexts": [["American Airlines allows one carry-on bag...", "For checked baggage, fees apply..."]]
    })
    
    # Calculate metrics
    results = {}
    results["faithfulness"] = faithfulness.compute(eval_dataset)
    results["answer_relevancy"] = answer_relevancy.compute(eval_dataset)
    results["context_relevancy"] = context_relevancy.compute(eval_dataset)
    results["harmfulness"] = harmfulness.compute(eval_dataset)
    
    # Custom evaluation with human feedback
    def evaluate_with_human_feedback(model_outputs, human_ratings):
        """
        model_outputs: List of model responses
        human_ratings: Dict with keys like 'accuracy', 'helpfulness' with 1-5 ratings
        """
        avg_rating = np.mean(list(human_ratings.values()))
        return {
            "average_rating": avg_rating,
            "accuracy_rating": human_ratings.get('accuracy', 0),
            "helpfulness_rating": human_ratings.get('helpfulness', 0)
        }

# =====================================================================
# 6. AZURE AI SERVICES
# =====================================================================

def azure_ai_services():
    """Azure AI services code snippets"""
    # Azure OpenAI
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    response = client.chat.completions.create(
        model="gpt-4",  # deployment name
        messages=[
            {"role": "system", "content": "You are an American Airlines assistant."},
            {"role": "user", "content": "Tell me about your loyalty program."}
        ]
    )
    
    # Azure AI Search (Vector Search)
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="aa-knowledge-base",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )
    
    # Vector search
    vector_query = search_client.search(
        search_text=None,
        vector=query_embedding,  # Vector from embedding model
        top_k=3,
        vector_fields="contentVector"
    )
    
    # Azure AI Document Intelligence (formerly Form Recognizer)
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    
    document_client = DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
    )
    
    # Extract text from boarding pass or ticket
    with open("boarding_pass.pdf", "rb") as f:
        poller = document_client.begin_analyze_document("prebuilt-document", f.read())
    
    result = poller.result()

# =====================================================================
# 7. MACHINE LEARNING FOR AIRLINE OPERATIONS
# =====================================================================

def airline_ml_models():
    """ML models for airline operations"""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
    
    # Example: Flight delay prediction
    def flight_delay_prediction(df):
        # Features: origin, destination, time of day, day of week, month, aircraft type, weather
        X = df.drop(['delay_minutes'], axis=1)
        y = df['delay_minutes']
        
        # Preprocessing
        numeric_features = ['distance', 'scheduled_time', 'aircraft_age']
        categorical_features = ['origin', 'destination', 'day_of_week', 'month', 'aircraft_type']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Model pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train and evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        return pipeline, mae
    
    # Example: Customer churn prediction
    def customer_churn_prediction(df):
        # Features: flight frequency, loyalty status, avg ticket price, complaint history, etc.
        X = df.drop(['churned'], axis=1)
        y = df['churned']
        
        # Similar preprocessing and pipeline setup
        # ...
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return best_model, accuracy, report

# =====================================================================
# 8. DATA VISUALIZATION
# =====================================================================

def airline_data_visualization():
    """Data visualization for airline data"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Matplotlib/Seaborn
    def basic_visualizations(df):
        # Delay distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['delay_minutes'], kde=True)
        plt.title('Distribution of Flight Delays')
        plt.xlabel('Delay (minutes)')
        plt.savefig('delay_distribution.png')
        
        # Delay by airport
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='origin', y='delay_minutes', data=df)
        plt.title('Delay by Origin Airport')
        plt.xticks(rotation=90)
        plt.savefig('delay_by_airport.png')
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=['number']).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Between Numerical Features')
        plt.savefig('correlation_heatmap.png')
    
    # Plotly (interactive)
    def interactive_visualizations(df):
        # Route map
        fig = px.scatter_geo(
            df, 
            lat='origin_lat', 
            lon='origin_lon',
            color='avg_delay',
            hover_name='origin',
            size='flight_count',
            projection='natural earth'
        )
        fig.update_layout(title='Flight Delays by Origin')
        fig.write_html('route_map.html')
        
        # Time series of delays
        fig = px.line(
            df.groupby('date')['delay_minutes'].mean().reset_index(),
            x='date',
            y='delay_minutes',
            title='Average Delay Over Time'
        )
        fig.write_html('delay_time_series.html')
        
        # Dashboard with multiple plots
        from dash import Dash, dcc, html
        import dash_bootstrap_components as dbc
        
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = dbc.Container([
            html.H1("American Airlines Operations Dashboard"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='delay-map', figure=fig), width=12)
            ])
            # Add more components as needed
        ])
        
        # Run with: app.run_server(debug=True)

print("Gen AI Data Scientist Cheatsheet for American Airlines Interview")