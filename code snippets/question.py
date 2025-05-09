comment = "My flight got canceled and I didn’t receive any assistance"

category_list = [
"IROPS HANDLING - Flight delays, cancellations, and disruptions",
"OPERATIONS - Ground operations, boarding experience",
"AADVANTAGE LOYALTY PROGRAM - PRODUCTS: Comments about AAdvantage account questions, loyalty status and perks, using benefits, mileage questions, issues with miles where miles from their trip did not post into their account"
]

few_shots = [
{"comment": "My flight got canceled and I didn’t receive any assistance.", "category": "IROPS HANDLING - Flight delays, cancellations, and disruptions"},
{"comment":  "I had issues boarding the plane with my wheelchair.", "category": "OPERATIONS - Ground operations, boarding experience"}
]


few_shot_prompt = [
    f"Comment: {few_shot['comment']}. Category: {few_shot['category']}" for few_shot in few_shots
]

system_message = (
    "You are an AI assistant specialized in categorizing American Airlines customer complaints. "
    "Your task is to analyze each customer comment and assign it to the most appropriate category. "
    "If the comment fits multiple categories, you may assign up to three categories, but preferably just one. "
    "You should also analyze the sentiment of each comment as Negative, Neutral, or Positive. "
    "Provide your response in a structured JSON format."
    )
