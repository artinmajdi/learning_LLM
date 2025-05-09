"""
FEATURE IMPORTANCE & NON-AZURE GEN AI TECHNIQUES
Supplementary Cheatsheet for American Airlines Interview
"""

# =====================================================================
# 1. ADDITIONAL FEATURE IMPORTANCE TECHNIQUES
# =====================================================================

# --- Scikit-learn Feature Importance ---
def sklearn_feature_importance():
    """Feature importance methods with scikit-learn"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.feature_selection import SelectFromModel, RFE, RFECV
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # 1. Tree-based feature importance
    def tree_based_importance(X_train, X_test, y_train, y_test):
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Built-in feature importance (Mean Decrease in Impurity)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Random Forest Feature Importances")
        plt.bar(range(X_train.shape[1]), importances[indices])
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.tight_layout()
        
        # Gradient Boosting importance
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        gb_importances = gb.feature_importances_
        
        return {"rf_importances": importances, "gb_importances": gb_importances}
    
    # 2. Permutation importance (model-agnostic)
    def get_permutation_importance(model, X_test, y_test):
        # More reliable than MDI for correlated features
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        # Plot
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        plt.figure(figsize=(10, 6))
        plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
        plt.title("Permutation Importances")
        plt.tight_layout()
        
        return perm_importance
    
    # 3. Feature selection methods
    def feature_selection_methods(X_train, y_train):
        # L1-based feature selection (Lasso)
        lasso = SelectFromModel(Lasso(alpha=0.01))
        lasso.fit(X_train, y_train)
        selected_features_lasso = X_train.columns[lasso.get_support()]
        
        # Recursive Feature Elimination
        rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
        rfe.fit(X_train, y_train)
        selected_features_rfe = X_train.columns[rfe.support_]
        
        # RFE with Cross-Validation
        rfecv = RFECV(estimator=LogisticRegression(), cv=5)
        rfecv.fit(X_train, y_train)
        selected_features_rfecv = X_train.columns[rfecv.support_]
        
        # Plot CV scores
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.title("Feature Selection with RFE-CV")
        
        return {
            "lasso_selected": selected_features_lasso,
            "rfe_selected": selected_features_rfe,
            "rfecv_selected": selected_features_rfecv,
            "optimal_feature_count": rfecv.n_features_
        }

# --- Eli5 for Model Inspection ---
def eli5_feature_importance():
    """Feature importance with eli5"""
    import eli5
    from eli5.sklearn import PermutationImportance
    from sklearn.ensemble import RandomForestClassifier
    from IPython.display import display
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Show model weights
    display(eli5.show_weights(model, feature_names=list(X_train.columns)))
    
    # Permutation importance
    perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
    display(eli5.show_weights(perm, feature_names=list(X_test.columns)))
    
    # Explain prediction for a single passenger
    display(eli5.show_prediction(model, X_test.iloc[0], feature_names=list(X_test.columns)))

# --- LIME for Local Explanations ---
def lime_explanations():
    """Local explanations with LIME"""
    import lime
    import lime.lime_tabular
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Create explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=["Not Satisfied", "Satisfied"],
        discretize_continuous=True
    )
    
    # Explain a prediction
    passenger_idx = 0
    exp = explainer.explain_instance(
        X_test.iloc[passenger_idx].values, 
        model.predict_proba,
        num_features=6
    )
    
    # Visualize explanation
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for Passenger {passenger_idx}")
    plt.tight_layout()
    
    # Show feature weights in text format
    exp.show_in_notebook(show_table=True)
    
    # For multiple instances
    def explain_multiple_passengers(model, X_test, explainer, n_passengers=5):
        explanations = []
        for i in range(min(n_passengers, len(X_test))):
            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                model.predict_proba,
                num_features=6
            )
            explanations.append(exp)
        return explanations

# --- Partial Dependence Plots ---
def partial_dependence_plots():
    """Partial dependence plots for feature effects"""
    from sklearn.inspection import plot_partial_dependence
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Features to analyze
    features = ["FlightDelayMinutes", "SeatComfort", "StaffAttitude"]
    
    # Plot PDPs
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_partial_dependence(
        model, X_train, features, 
        n_jobs=3, grid_resolution=50, ax=ax
    )
    plt.suptitle("Partial Dependence of Customer Satisfaction")
    plt.tight_layout()
    
    # 2D partial dependence (interaction)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_partial_dependence(
        model, X_train, 
        [(0, 1)],  # Interaction between first two features
        n_jobs=3, grid_resolution=50, ax=ax
    )
    plt.suptitle("2D Partial Dependence")
    plt.tight_layout()

# =====================================================================
# 2. NON-AZURE VECTOR DATABASES
# =====================================================================

def vector_database_examples():
    """Examples of non-Azure vector databases"""
    import os
    import numpy as np
    from langchain_openai import OpenAIEmbeddings
    
    # --- Pinecone ---
    def pinecone_example():
        import pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        # Create or connect to index
        index_name = "american-airlines-docs"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        # Connect to index
        index = pinecone.Index(index_name)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
        
        # Add documents (if needed)
        # vector_store.add_documents(documents)
        
        # Create retriever and QA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        return qa_chain
    
    # --- Weaviate ---
    def weaviate_example():
        import weaviate
        from langchain_weaviate import WeaviateVectorStore
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        
        # Connect to Weaviate
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        
        # Create schema if it doesn't exist
        if not client.schema.exists("AADocument"):
            class_obj = {
                "class": "AADocument",
                "vectorizer": "none",  # Using external vectorizer
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "source", "dataType": ["string"]},
                    {"name": "metadata", "dataType": ["string"]}
                ]
            }
            client.schema.create_class(class_obj)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = WeaviateVectorStore(
            client=client,
            index_name="AADocument",
            text_key="content",
            embedding=embeddings
        )
        
        # Create retriever and QA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        return qa_chain
    
    # --- Milvus ---
    def milvus_example():
        from pymilvus import connections, Collection, utility
        from langchain_community.vectorstores import Milvus
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530")
        )
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Milvus(
            collection_name="american_airlines_docs",
            embedding_function=embeddings,
            connection_args={"host": os.getenv("MILVUS_HOST", "localhost"), "port": os.getenv("MILVUS_PORT", "19530")}
        )
        
        # Create retriever and QA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        return qa_chain

# =====================================================================
# 3. OPEN SOURCE LLMs
# =====================================================================

def open_source_llms():
    """Examples of using open source LLMs"""
    
    # --- Hugging Face Models with LangChain ---
    def huggingface_models():
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        
        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load LLM
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain HF pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create vector store and retriever
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        return qa_chain
    
    # --- LlamaCpp for Local Inference ---
    def llamacpp_example():
        from langchain_community.llms import LlamaCpp
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load LLM
        llm = LlamaCpp(
            model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            temperature=0.7,
            max_tokens=2000,
            n_ctx=4096,
            top_p=0.95,
            verbose=True
        )
        
        # Create vector store and retriever
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        return qa_chain
    
    # --- Ollama for Local API ---
    def ollama_example():
        from langchain_community.llms import Ollama
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load LLM
        llm = Ollama(
            model="mistral",
            temperature=0.7,
            verbose=True
        )
        
        # Create vector store and retriever
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        return qa_chain

# =====================================================================
# 4. ADVANCED FEATURE ENGINEERING
# =====================================================================

def advanced_feature_engineering():
    """Advanced feature engineering techniques"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    
    # --- Automated Feature Engineering ---
    def automated_feature_engineering(df):
        import featuretools as ft
        
        # Create entity set
        es = ft.EntitySet(id="aa_data")
        
        # Add dataframes
        es.add_dataframe(
            dataframe_name="flights",
            dataframe=df[["flight_id", "origin", "destination", "departure_time", "arrival_time"]],
            index="flight_id"
        )
        
        es.add_dataframe(
            dataframe_name="passengers",
            dataframe=df[["passenger_id", "flight_id", "age", "loyalty_status"]],
            index="passenger_id"
        )
        
        # Define relationships
        r = ft.Relationship(es["flights"]["flight_id"], es["passengers"]["flight_id"])
        es.add_relationship(r)
        
        # Run deep feature synthesis
        features, feature_names = ft.dfs(
            entityset=es,
            target_dataframe_name="passengers",
            agg_primitives=["mean", "sum", "std", "count"],
            trans_primitives=["day", "month", "year", "weekday"]
        )
        
        return features, feature_names
    
    # --- Time Series Feature Extraction ---
    def time_series_features(df, date_col="departure_date"):
        import tsfresh
        from tsfresh.feature_extraction import extract_features
        from tsfresh.feature_selection import select_features
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract basic time features
        df["hour"] = df[date_col].dt.hour
        df["day"] = df[date_col].dt.day
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["quarter"] = df[date_col].dt.quarter
        
        # For time series data (e.g., flight delays by date)
        # Prepare data in required format for tsfresh
        ts_data = df[["flight_id", date_col, "delay_minutes"]].copy()
        ts_data.rename(columns={date_col: "time", "flight_id": "id"}, inplace=True)
        
        # Extract features
        extracted_features = extract_features(ts_data, column_id="id", column_sort="time")
        
        # Select relevant features
        if "target" in df.columns:
            selected_features = select_features(extracted_features, df["target"])
            return selected_features
        
        return extracted_features
    
    # --- Advanced Preprocessing Pipeline ---
    def advanced_preprocessing_pipeline(df):
        # Define column types
        numeric_features = ["age", "flight_duration", "ticket_price", "previous_flights"]
        categorical_features = ["origin", "destination", "aircraft_type", "loyalty_status"]
        date_features = ["booking_date", "departure_date"]
        
        # Numeric transformers
        numeric_transformer = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", PowerTransformer(method="yeo-johnson"))  # Better than StandardScaler for skewed data
        ])
        
        # Categorical transformers
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        # Date transformers
        def extract_date_features(X):
            X = X.copy()
            for col in X.columns:
                X[col] = pd.to_datetime(X[col])
                X[f"{col}_month"] = X[col].dt.month
                X[f"{col}_day"] = X[col].dt.day
                X[f"{col}_dayofweek"] = X[col].dt.dayofweek
                X[f"{col}_is_weekend"] = X[col].dt.dayofweek.isin([5, 6]).astype(int)
            return X
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
        
        # Create pipeline with feature selection
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("variance_filter", VarianceThreshold(threshold=0.01))  # Remove low-variance features
        ])
        
        # Transform data
        X_transformed = pipeline.fit_transform(df[numeric_features + categorical_features])
        
        # Handle date features separately
        if date_features:
            date_df = extract_date_features(df[date_features])
            # Combine with transformed features
            # (would need to convert X_transformed to DataFrame with proper column names)
        
        return X_transformed, pipeline

print("Feature Importance & Non-Azure Gen AI Techniques Cheatsheet")