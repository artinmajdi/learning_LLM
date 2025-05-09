"""
GenAI Classifier for American Airlines Customer Complaints
This module implements a prompt-engineering based classifier that categorizes
customer complaints into predefined categories using few-shot learning techniques.
"""

# Define the complaint categories
category_list = [
    "IROPS HANDLING - Flight delays, cancellations, and disruptions",
    "OPERATIONS - Ground operations, boarding experience",
    "AADVANTAGE LOYALTY PROGRAM - PRODUCTS: Comments about AAdvantage account questions, loyalty status and perks, using benefits, mileage questions, issues with miles where miles from their trip did not post into their account"
]

# Define few-shot examples
few_shots = [
    {"comment": "My flight got canceled and I didn't receive any assistance.", "category": "IROPS HANDLING - Flight delays, cancellations, and disruptions"},
    {"comment": "I had issues boarding the plane with my wheelchair.", "category": "OPERATIONS - Ground operations, boarding experience"}
]

# Question 1: Generate the formatted few-shot prompt string
def format_few_shot_examples(examples):
    """
    Format the few-shot examples into a prompt-friendly string.

    Args:
        examples: List of dictionaries containing comment and category pairs

    Returns:
        List of formatted strings for few-shot learning
    """
    few_shot_prompt = []

    for example in examples:
        formatted_example = f"Comment: {example['comment']} Category: {example['category']}"
        few_shot_prompt.append(formatted_example)

    return few_shot_prompt

# Generate the few-shot prompt
few_shot_prompt = format_few_shot_examples(few_shots)

# Question 2.1: Define the system message
system_message = (
    "You are an AI assistant specialized in categorizing American Airlines customer complaints. "
    "Your task is to analyze each customer comment and assign it to the most appropriate category. "
    "If the comment fits multiple categories, you may assign up to three categories, but preferably just one. "
    "You should also analyze the sentiment of each comment as Negative, Neutral, or Positive. "
    "Provide your response in a structured JSON format."
)

# Question 2.2: Implement the generate_prompt function
def generate_prompt(comment, categories, few_shot_examples):
    """
    Generate a well-structured prompt for the GPT model to classify customer complaints.

    Args:
        comment: The customer comment to be classified
        categories: List of valid categories with descriptions
        few_shot_examples: List of formatted few-shot examples

    Returns:
        A complete prompt for GPT classification
    """
    # Convert few-shot examples list to a string with line breaks
    few_shot_section = "\n".join(few_shot_examples)

    # Format the categories as a numbered list
    categories_section = "\n".join([f"{i+1}. {category}" for i, category in enumerate(categories)])

    prompt = f"""
        Please classify the following customer comment into the most appropriate category from the list below.
        If the comment fits multiple categories, you may assign up to three categories, but preferably just one.
        Also, analyze the sentiment of the comment as Negative, Neutral, or Positive.

        VALID CATEGORIES:
        {categories_section}

        FEW-SHOT EXAMPLES:
        {few_shot_section}

        COMMENT TO CLASSIFY:
        {comment}

        OUTPUT FORMAT:
        Provide your response as a minified JSON object with the following structure:
        {{
            "categories": ["Primary Category", "Secondary Category (if applicable)"],
            "sentiment": "Negative/Neutral/Positive"
        }}
        """
    return prompt

# Question 2.3: Print a sample prompt
def print_sample_prompt():
    """Print a sample prompt for demonstration purposes"""
    sample_comment = "My flight got canceled and I didn't receive any assistance"
    sample_prompt = generate_prompt(sample_comment, category_list, few_shot_prompt)
    print(sample_prompt)

    return sample_prompt

# Execute the sample prompt generation
if __name__ == "__main__":
    print("Formatted Few-Shot Examples:")
    for example in few_shot_prompt:
        print(f"- {example}")

    print("\nSystem Message:")
    print(system_message)

    print("\nSample Prompt:")
    print_sample_prompt()
