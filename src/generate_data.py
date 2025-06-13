import json
import random
from pathlib import Path

# Define categories and their example scenarios
CATEGORIES = {
    "Technical": [
        "My laptop won't turn on after the latest update",
        "Getting error code 404 when trying to access the dashboard",
        "Software keeps crashing when I try to save files",
        "Can't connect to the WiFi network",
        "Printer is showing offline status",
        "System is running very slow after update",
        "Getting blue screen error frequently",
        "Application freezes when uploading large files",
        "Can't install the new software version",
        "Hardware compatibility issues with new device"
    ],
    "Billing": [
        "I was charged twice for my subscription",
        "Need to update my payment method",
        "When will I receive my refund?",
        "Why was I charged extra fees?",
        "Need to cancel my subscription",
        "Billing cycle change request",
        "Invoice not received for last month",
        "Payment declined error",
        "Need to change my billing address",
        "Subscription renewal questions"
    ],
    "Account": [
        "Can't log in to my account",
        "Need to reset my password",
        "Account locked due to multiple failed attempts",
        "Want to change my email address",
        "Two-factor authentication not working",
        "Need to update account information",
        "Account access issues from new device",
        "Profile settings not saving",
        "Account verification problems",
        "Need to merge multiple accounts"
    ],
    "Product": [
        "How do I use the new feature?",
        "Product comparison questions",
        "Feature request for next update",
        "Product compatibility questions",
        "Need help with product setup",
        "Product documentation request",
        "Feature availability in my region",
        "Product roadmap questions",
        "Integration with other tools",
        "Product limitations and capabilities"
    ],
    "Other": [
        "General feedback about service",
        "Partnership inquiries",
        "Company information request",
        "Office location questions",
        "Career opportunities",
        "Press and media inquiries",
        "Event participation queries",
        "Community guidelines questions",
        "Sustainability initiatives",
        "Company policies clarification"
    ]
}

def generate_training_data(num_examples=50):
    """Generate training data with balanced categories."""
    examples = []
    examples_per_category = num_examples // len(CATEGORIES)
    
    for category, scenarios in CATEGORIES.items():
        for _ in range(examples_per_category):
            # Randomly select a base scenario
            base_scenario = random.choice(scenarios)
            
            # Create variations of the base scenario
            variations = [
                base_scenario,
                f"I'm having an issue: {base_scenario.lower()}",
                f"Help needed: {base_scenario.lower()}",
                f"Question about {base_scenario.lower()}",
                f"Support request: {base_scenario.lower()}"
            ]
            
            # Add some random details
            details = [
                "This is urgent.",
                "Can someone help me with this?",
                "I've tried everything I can think of.",
                "This is affecting my work.",
                "Need immediate assistance."
            ]
            
            # Create the final message
            message = f"{random.choice(variations)} {random.choice(details)}"
            
            # Create the training example in the correct format for chat completion fine-tuning
            example = {
                "messages": [
                    {"role": "system", "content": "You are a customer support ticket classifier. Respond with only one of these categories: Technical, Billing, Account, Product, or Other."},
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": category}
                ]
            }
            examples.append(example)
    
    return examples

def generate_test_data(num_examples=15):
    """Generate test data with balanced categories."""
    examples = []
    examples_per_category = num_examples // len(CATEGORIES)
    
    for category, scenarios in CATEGORIES.items():
        for _ in range(examples_per_category):
            # Create more complex test scenarios
            base_scenario = random.choice(scenarios)
            message = f"I'm experiencing an issue with {base_scenario.lower()}. " \
                     f"This is causing problems with my work and I need immediate assistance. " \
                     f"Can you please help me resolve this?"
            
            # Create the test example in the correct format for chat completion fine-tuning
            example = {
                "messages": [
                    {"role": "system", "content": "You are a customer support ticket classifier. Respond with only one of these categories: Technical, Billing, Account, Product, or Other."},
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": category}
                ]
            }
            examples.append(example)
    
    return examples

def save_data(data, filename):
    """Save data to a JSONL file."""
    with open(filename, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')

def main():
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Generate and save training data
    training_data = generate_training_data(50)
    save_data(training_data, "data/training_data.jsonl")
    print(f"Generated {len(training_data)} training examples")
    
    # Generate and save test data
    test_data = generate_test_data(15)
    save_data(test_data, "data/test_data.jsonl")
    print(f"Generated {len(test_data)} test examples")

if __name__ == "__main__":
    main() 