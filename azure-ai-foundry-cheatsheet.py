"""
AZURE AI FOUNDRY INTERVIEW CHEAT SHEET FOR AMERICAN AIRLINES
==========================================================

This file contains common interview questions and Python code examples
for a Gen AI Data Scientist position focused on Azure AI Foundry.
"""

#-----------------------------------------------------------------------------
# 1. CORE CONCEPTS - AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: What is Azure AI Foundry and how does it differ from Azure OpenAI Service?

A: Azure AI Foundry is Microsoft's comprehensive platform for building, deploying, 
and managing AI models at scale with enterprise-grade security and governance. 
While Azure OpenAI Service focuses on providing access to OpenAI models,
AI Foundry adds capabilities like:
- Unified model lifecycle management
- Fine-tuning and customization tools
- Prompt engineering interfaces
- Specialized deployment options
- Integrated monitoring and observability
- Industry-specific solution accelerators

Azure OpenAI Service is actually a component within the broader Azure AI Foundry ecosystem.
"""

#-----------------------------------------------------------------------------
# 2. MODEL DEPLOYMENT WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you deploy a fine-tuned model from Azure AI Foundry to production?
"""

from azure.ai.foundry import AIFoundryClient
from azure.identity import DefaultAzureCredential

# Authenticate to Azure
credential = DefaultAzureCredential()
client = AIFoundryClient(credential=credential)

# Deploy a fine-tuned model
def deploy_model(model_id, deployment_name, capacity):
    """Deploy a fine-tuned model to an endpoint"""
    deployment = client.deployments.create_or_update(
        model_id=model_id,
        deployment_name=deployment_name,
        capacity=capacity,
        scaling_type="Standard"
    )
    return deployment

# Example usage
model_id = "ft:gpt-35-turbo:american-airlines:flight-ops:2025-05-01"
deployment = deploy_model(
    model_id=model_id,
    deployment_name="flight-operations-assistant",
    capacity=1  # Number of inference units
)

print(f"Model deployed to: {deployment.endpoint_url}")


#-----------------------------------------------------------------------------
# 3. PROMPT ENGINEERING WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you structure a prompt for a customer service chatbot that needs
to handle flight booking, status updates, and common airline questions?
"""

import openai
from azure.identity import DefaultAzureCredential
from azure.ai.foundry.prompts import PromptTemplate

# Setup API connection
client = openai.AzureOpenAI(
    azure_endpoint="https://aa-ai-foundry.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2023-12-01-preview"
)

# Create a system prompt template
system_prompt_template = """
You are an American Airlines customer service assistant. Your role is to help customers with:
1. Flight bookings and modifications
2. Flight status updates
3. Baggage information
4. Frequent flyer program questions

Use these guidelines:
- Be courteous and professional
- Provide concise, accurate information
- For complex requests, guide customers to appropriate self-service tools
- Adhere to American Airlines' policies regarding {policy_area}
- If you cannot help with a specific issue, offer to connect them with a human agent

Today's date is: {current_date}
"""

# Create prompt template with parameters
prompt = PromptTemplate(
    template=system_prompt_template,
    parameters=["policy_area", "current_date"]
)

# Generate a specific prompt
customer_prompt = prompt.format(
    policy_area="flight changes and cancellations",
    current_date="2025-05-09"
)

# Use the prompt
response = client.chat.completions.create(
    model="flight-operations-assistant",  # Our deployed model
    messages=[
        {"role": "system", "content": customer_prompt},
        {"role": "user", "content": "I need to change my flight tomorrow due to weather concerns."}
    ],
    temperature=0.3  # Lower temperature for more consistent responses
)

print(response.choices[0].message.content)


#-----------------------------------------------------------------------------
# 4. FINE-TUNING MODELS IN AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you prepare a dataset and fine-tune a model in Azure AI Foundry
for an airline-specific task?
"""

import pandas as pd
from azure.ai.foundry import AIFoundryClient
from azure.identity import DefaultAzureCredential

# 1. Prepare your fine-tuning dataset
def prepare_finetune_data():
    """Prepare fine-tuning dataset for flight operations assistant"""
    # Load proprietary airline data
    df = pd.read_csv("flight_operations_qa.csv")
    
    # Format into required structure for fine-tuning
    fine_tune_data = []
    for _, row in df.iterrows():
        fine_tune_data.append({
            "messages": [
                {"role": "system", "content": "You are a flight operations assistant for American Airlines."},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        })
    
    # Save to JSONL format as required by Azure AI Foundry
    import json
    with open("flight_ops_finetune.jsonl", "w") as f:
        for entry in fine_tune_data:
            f.write(json.dumps(entry) + "\n")
    
    return "flight_ops_finetune.jsonl"

# 2. Upload and fine-tune the model
def finetune_model():
    """Fine-tune a model with airline-specific data"""
    # Prepare the data
    training_file = prepare_finetune_data()
    
    # Connect to Azure AI Foundry
    credential = DefaultAzureCredential()
    client = AIFoundryClient(credential=credential)
    
    # Upload training file
    training_file_id = client.files.upload(
        file=training_file,
        purpose="fine-tune"
    )
    
    # Create fine-tuning job
    fine_tuning_job = client.fine_tuning.create(
        model="gpt-35-turbo",  # Base model to fine-tune
        training_file=training_file_id,
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 8,
            "learning_rate_multiplier": 2.0
        },
        suffix="flight-ops"  # Added to the model name
    )
    
    # Check fine-tuning status
    job_status = client.fine_tuning.retrieve(fine_tuning_job.id)
    print(f"Fine-tuning status: {job_status.status}")
    
    return fine_tuning_job.id


#-----------------------------------------------------------------------------
# 5. DATA INTEGRATION WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you integrate real-time flight data with Azure AI Foundry models?
"""

import pandas as pd
import numpy as np
from azure.identity import DefaultAzureCredential
from azure.ai.foundry import AIFoundryClient
import azure.functions as func
import json

# Sample function app that processes flight data and sends to AI model
def process_flight_data(flight_data_blob: func.InputStream):
    """Azure Function to process incoming flight data and send to AI model"""
    # Read incoming flight data
    flight_data = pd.read_json(flight_data_blob.read())
    
    # Preprocess data
    processed_data = preprocess_flight_data(flight_data)
    
    # Send to Azure AI Foundry for analysis
    predictions = send_to_ai_model(processed_data)
    
    # Return predictions
    return func.HttpResponse(
        json.dumps({"predictions": predictions}),
        mimetype="application/json"
    )

def preprocess_flight_data(flight_data):
    """Preprocess flight data for model consumption"""
    # Feature engineering
    flight_data['delay_risk'] = np.where(
        (flight_data['weather_condition'] == 'severe') | 
        (flight_data['maintenance_status'] == 'pending'),
        'high', 'low'
    )
    
    # Normalize numerical features
    numerical_cols = ['scheduled_duration', 'aircraft_age', 'passenger_load']
    flight_data[numerical_cols] = (flight_data[numerical_cols] - 
                                flight_data[numerical_cols].mean()) / flight_data[numerical_cols].std()
    
    return flight_data

def send_to_ai_model(processed_data):
    """Send processed data to AI Foundry model"""
    # Connect to Azure AI Foundry
    credential = DefaultAzureCredential()
    client = AIFoundryClient(credential=credential)
    
    # Call the deployed model
    response = client.chat.completions.create(
        model="flight-operations-assistant",
        messages=[
            {"role": "system", "content": "Analyze flight data and predict operational issues."},
            {"role": "user", "content": f"Flight data: {processed_data.to_json()}"}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content


#-----------------------------------------------------------------------------
# 6. RESPONSIBLE AI WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you implement responsible AI practices for a customer-facing
application using Azure AI Foundry?
"""

from azure.ai.foundry import AIFoundryClient
from azure.ai.foundry.responsible_ai import ResponsibleAIClient
from azure.identity import DefaultAzureCredential

def implement_responsible_ai(model_id):
    """Implement Responsible AI practices for a customer-facing model"""
    # Connect to Azure AI Foundry
    credential = DefaultAzureCredential()
    client = AIFoundryClient(credential=credential)
    
    # Get Responsible AI client
    rai_client = ResponsibleAIClient(credential=credential)
    
    # 1. Set up content filtering
    content_filter = client.filters.create_or_update(
        name="airline_content_filter",
        categories={
            "hate": "block",
            "sexual": "block",
            "violence": "block",
            "self_harm": "block",
            "protected_categories": "warn"
        }
    )
    
    # 2. Apply content filter to model
    client.deployments.update(
        model_id=model_id,
        content_filter_id=content_filter.id
    )
    
    # 3. Set up monitoring for bias detection
    monitoring = rai_client.monitoring.create(
        model_id=model_id,
        settings={
            "bias_detection": {
                "protected_attributes": ["nationality", "gender", "age"],
                "threshold": 0.05,
                "alert_on_threshold_exceeded": True
            },
            "data_drift": {
                "enabled": True,
                "threshold": 0.1
            }
        }
    )
    
    # 4. Create transparency note
    transparency = rai_client.transparency.create(
        model_id=model_id,
        note={
            "purpose": "Customer service automation for American Airlines",
            "limitations": "The model may not have real-time information on flight changes",
            "data_sources": ["Historical customer interactions", "Flight operations data"],
            "monitoring_practices": "Continuous evaluation for bias and fairness"
        }
    )
    
    return {
        "content_filter": content_filter.id,
        "monitoring": monitoring.id,
        "transparency": transparency.id
    }


#-----------------------------------------------------------------------------
# 7. PREDICTIVE MAINTENANCE WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you implement a predictive maintenance solution for aircraft
components using Azure AI Foundry?
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from azure.ai.foundry import AIFoundryClient
from azure.identity import DefaultAzureCredential
import mlflow
import mlflow.sklearn

def train_maintenance_model():
    """Train a predictive maintenance model for aircraft components"""
    # Load historical maintenance data
    maintenance_data = pd.read_csv("aircraft_maintenance_history.csv")
    
    # Feature engineering
    maintenance_data['days_since_last_maintenance'] = (
        pd.to_datetime('today') - pd.to_datetime(maintenance_data['last_maintenance_date'])
    ).dt.days
    
    # Create features and target
    features = ['component_age', 'flight_hours', 'cycle_count', 
                'days_since_last_maintenance', 'vibration_reading',
                'temperature_reading', 'pressure_reading']
    
    X = maintenance_data[features]
    y = maintenance_data['failure_within_30_days']  # Binary target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Log model with MLflow
    mlflow.set_tracking_uri("azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/...")
    mlflow.set_experiment("aircraft-predictive-maintenance")
    
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 100,
            "features": features
        })
        mlflow.log_metrics({"accuracy": accuracy})
        mlflow.sklearn.log_model(model, "maintenance_prediction_model")
    
    return model

# Function to deploy the model to Azure AI Foundry
def deploy_maintenance_model(model_uri):
    """Deploy the maintenance model to Azure AI Foundry"""
    credential = DefaultAzureCredential()
    client = AIFoundryClient(credential=credential)
    
    # Register the model
    registered_model = client.models.create_or_update(
        name="aircraft-maintenance-predictor",
        source=model_uri,
        description="Predictive maintenance model for aircraft components"
    )
    
    # Deploy the model
    endpoint = client.endpoints.create_or_update(
        name="maintenance-prediction-endpoint",
        auth_mode="key"
    )
    
    deployment = client.deployments.create_or_update(
        endpoint_name=endpoint.name,
        name="maintenance-predictor",
        model_id=registered_model.id,
        instance_type="Standard_DS3_v2",
        instance_count=1
    )
    
    print(f"Model deployed to: {endpoint.endpoint_url}")
    return endpoint.endpoint_url


#-----------------------------------------------------------------------------
# (8) CUSTOMER SENTIMENT ANALYSIS WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you implement a customer sentiment analysis system for 
American Airlines customer feedback using Azure AI Foundry?
"""

import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.foundry import AIFoundryClient
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def analyze_customer_feedback(feedback_data):
    """Analyze customer feedback using Azure AI services"""
    # Connect to Azure Text Analytics
    credential = DefaultAzureCredential()
    text_analytics_client = TextAnalyticsClient(
        endpoint="https://aa-text-analytics.cognitiveservices.azure.com/",
        credential=credential
    )
    
    # Connect to Azure AI Foundry
    ai_foundry_client = AIFoundryClient(credential=credential)
    
    # Extract feedback text from data
    feedback_texts = feedback_data['feedback_text'].tolist()
    
    # 1. Sentiment analysis with Text Analytics
    sentiment_response = text_analytics_client.analyze_sentiment(
        documents=feedback_texts, 
        show_opinion_mining=True
    )
    
    # Process sentiment results
    sentiments = []
    for doc in sentiment_response:
        if doc.is_error:
            sentiments.append(None)
        else:
            sentiments.append({
                'sentiment': doc.sentiment,
                'positive_score': doc.confidence_scores.positive,
                'neutral_score': doc.confidence_scores.neutral,
                'negative_score': doc.confidence_scores.negative
            })
    
    # Add sentiment results to dataframe
    feedback_data['sentiment_results'] = sentiments
    
    # 2. Key phrase extraction for deeper analysis
    keyphrase_response = text_analytics_client.extract_key_phrases(
        documents=feedback_texts
    )
    
    # Process key phrases
    keyphrases = []
    for doc in keyphrase_response:
        if doc.is_error:
            keyphrases.append([])
        else:
            keyphrases.append(doc.key_phrases)
    
    # Add key phrases to dataframe
    feedback_data['key_phrases'] = keyphrases
    
    # 3. More advanced analysis with Azure AI Foundry
    topic_clusters = ai_foundry_client.text.analyze_topics(
        documents=feedback_texts,
        num_topics=5
    )
    
    # Visualize sentiment distribution
    sentiment_counts = feedback_data['sentiment_results'].apply(
        lambda x: x['sentiment'] if x else 'unknown'
    ).value_counts()
    
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red', 'black'])
    plt.title('Customer Feedback Sentiment Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    plt.savefig('sentiment_distribution.png')
    
    # Generate word cloud of key phrases
    all_phrases = [phrase for sublist in keyphrases for phrase in sublist]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_phrases))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Key Topics in Customer Feedback')
    plt.savefig('feedback_wordcloud.png')
    
    return feedback_data


#-----------------------------------------------------------------------------
# 9. GENERATIVE AI FOR CONTENT CREATION WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you use generative AI in Azure AI Foundry to automatically 
create personalized content for American Airlines customers?
"""

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.ai.foundry import AIFoundryClient
import json

def generate_personalized_content(customer_data):
    """Generate personalized marketing content for airline customers"""
    # Connect to Azure AI Foundry
    credential = DefaultAzureCredential()
    client = AIFoundryClient(credential=credential)
    
    # Prepare customer segments
    customer_data['segment'] = pd.cut(
        customer_data['loyalty_points'], 
        bins=[0, 10000, 50000, 100000, float('inf')],
        labels=['basic', 'silver', 'gold', 'platinum']
    )
    
    # Generate personalized content for each customer
    personalized_content = []
    
    for _, customer in customer_data.iterrows():
        # Create customer profile for personalization
        profile = {
            "segment": customer['segment'],
            "frequently_visited": customer['top_destinations'],
            "travel_preferences": customer['preferences'],
            "upcoming_trips": customer['upcoming_flights'],
            "last_interaction": customer['last_contact_reason']
        }
        
        # Determine content type based on customer status
        if customer['days_since_last_flight'] > 90:
            content_type = "re-engagement"
        elif customer['upcoming_flights']:
            content_type = "trip-preparation"
        else:
            content_type = "destination-inspiration"
        
        # Generate the personalized content
        response = client.chat.completions.create(
            model="marketing-content-generation",  # A fine-tuned model
            messages=[
                {"role": "system", "content": f"""
                You are American Airlines' personalized marketing content creator.
                Generate {content_type} content for a {customer['segment']} tier customer.
                Be specific to their travel preferences and history.
                Include a clear call-to-action appropriate for this customer.
                Content should be concise, engaging, and on-brand.
                """},
                {"role": "user", "content": f"Customer profile: {json.dumps(profile)}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Store the generated content
        personalized_content.append({
            "customer_id": customer['customer_id'],
            "segment": customer['segment'],
            "content_type": content_type,
            "generated_content": response.choices[0].message.content,
            "delivery_channel": determine_preferred_channel(customer)
        })
    
    return personalized_content

def determine_preferred_channel(customer):
    """Determine the best channel to deliver content to this customer"""
    # Simple logic to determine preferred channel
    if customer['email_open_rate'] > 0.3:
        return "email"
    elif customer['app_user'] and customer['app_login_last_30_days'] > 0:
        return "app_notification"
    elif customer['sms_opt_in']:
        return "sms"
    else:
        return "email"  # Default fallback


#-----------------------------------------------------------------------------
# 10. HANDLING REAL-TIME QUERIES WITH AZURE AI FOUNDRY
#-----------------------------------------------------------------------------

"""
Q: How would you implement a system to handle real-time customer queries
about flight status using Azure AI Foundry?
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from azure.identity import DefaultAzureCredential
from azure.ai.foundry import AIFoundryClient
import datetime
import httpx

# Define API models
class FlightQuery(BaseModel):
    flight_number: str
    date: str = None  # Optional, defaults to today if not provided
    customer_id: str = None  # Optional for personalization

class FlightResponse(BaseModel):
    flight_number: str
    status: str
    departure_time: str
    arrival_time: str
    gate_info: str
    additional_info: str = None

# Create FastAPI app
app = FastAPI(title="American Airlines Flight Status API")

# Initialize Azure AI Foundry client
credential = DefaultAzureCredential()
ai_client = AIFoundryClient(credential=credential)

# Flight status endpoint
@app.post("/flight-status", response_model=FlightResponse)
async def get_flight_status(query: FlightQuery):
    try:
        # Set default date to today if not provided
        if not query.date:
            query.date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # 1. Get flight data from operational database
        flight_data = await get_flight_data_from_db(query.flight_number, query.date)
        
        # Handle case where flight is not found
        if not flight_data:
            raise HTTPException(status_code=404, detail=f"Flight {query.flight_number} not found")
        
        # 2. Check for weather or operational impacts (separate service)
        operational_impacts = await check_operational_impacts(
            departure_airport=flight_data["departure_airport"],
            arrival_airport=flight_data["arrival_airport"],
            scheduled_time=flight_data["scheduled_departure"]
        )
        
        # 3. Use Azure AI Foundry to generate a helpful response
        ai_response = ai_client.chat.completions.create(
            model="flight-status-assistant",
            messages=[
                {"role": "system", "content": """
                You are American Airlines' flight status assistant.
                Provide concise, helpful information about flight status.
                Include relevant details about gates, times, and any operational impacts.
                Format times in a user-friendly way.
                If there are delays or issues, provide context and options.
                """},
                {"role": "user", "content": f"""
                Generate a helpful flight status response based on this data:
                
                Flight: {flight_data}
                Operational Impacts: {operational_impacts}
                Customer ID: {query.customer_id if query.customer_id else 'Not provided'}
                """}
            ],
            temperature=0.2,
            max_tokens=200
        )
        
        # Extract generated message
        additional_info = ai_response.choices[0].message.content
        
        # 4. Create and return response
        return FlightResponse(
            flight_number=query.flight_number,
            status=flight_data["status"],
            departure_time=flight_data["scheduled_departure"],
            arrival_time=flight_data["scheduled_arrival"],
            gate_info=f"Departure: {flight_data['departure_gate']} - Arrival: {flight_data['arrival_gate']}",
            additional_info=additional_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def get_flight_data_from_db(flight_number, date):
    """Get flight data from operational database (mocked for this example)"""
    # In a real implementation, this would query your flight operations database
    # Mocked response for illustration
    return {
        "flight_number": flight_number,
        "status": "On Time",
        "scheduled_departure": "2025-05-09T14:30:00Z",
        "scheduled_arrival": "2025-05-09T16:45:00Z",
        "departure_airport": "DFW",
        "arrival_airport": "LAX",
        "departure_gate": "C15",
        "arrival_gate": "45B"
    }

async def check_operational_impacts(departure_airport, arrival_airport, scheduled_time):
    """Check for weather or operational impacts affecting the flight"""
    # In a real implementation, this would call weather and operations APIs
    # Mocked response for illustration
    return {
        "departure_weather": "Clear",
        "arrival_weather": "Light Rain",
        "departure_airport_status": "Normal Operations",
        "arrival_airport_status": "Minor Delays",
        "impact_level": "Low"
    }

# Run the API (for development)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
