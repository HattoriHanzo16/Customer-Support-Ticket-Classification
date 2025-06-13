import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_test_data(file_path):
    """Load test data from JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def get_baseline_prediction(message):
    """Get prediction from the baseline model (GPT-3.5-turbo)."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a customer support ticket classifier. Respond with only one of these categories: Technical, Billing, Account, Product, or Other."},
            {"role": "user", "content": message}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def get_fine_tuned_prediction(message, model_name):
    """Get prediction from the fine-tuned model."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a customer support ticket classifier. Respond with only one of these categories: Technical, Billing, Account, Product, or Other."},
            {"role": "user", "content": message}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def evaluate_model(examples, model_type="baseline"):
    """Evaluate model performance on test data."""
    true_labels = []
    predicted_labels = []
    
    print(f"\nEvaluating {model_type} model...")
    for example in examples:
        # Extract message and true label from the messages array
        message = example["messages"][1]["content"]  # User message
        true_label = example["messages"][2]["content"]  # Assistant message (true label)
        
        if model_type == "baseline":
            predicted_label = get_baseline_prediction(message)
        else:
            predicted_label = get_fine_tuned_prediction(message, model_type)
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        
        print(f"\nMessage: {message}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n{model_type.upper()} Model Accuracy: {accuracy:.2f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate customer support ticket classification models")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation only")
    args = parser.parse_args()
    
    # Load test data
    test_file = Path("data/test_data.jsonl")
    if not test_file.exists():
        print("Test data not found. Please run generate_data.py first.")
        return
    
    examples = load_test_data(test_file)
    
    # Evaluate baseline model
    baseline_accuracy = evaluate_model(examples, "baseline")
    
    if not args.baseline:
        # Check if fine-tuned model exists
        model_file = Path("data/fine_tuned_model.txt")
        if not model_file.exists():
            print("\nFine-tuned model not found. Please run fine_tune.py first.")
            return
        
        # Load fine-tuned model name
        with open(model_file, "r") as f:
            fine_tuned_model = f.read().strip()
        
        # Evaluate fine-tuned model
        fine_tuned_accuracy = evaluate_model(examples, fine_tuned_model)
        
        # Compare results
        improvement = fine_tuned_accuracy - baseline_accuracy
        print(f"\nImprovement over baseline: {improvement:.2%}")

if __name__ == "__main__":
    main() 