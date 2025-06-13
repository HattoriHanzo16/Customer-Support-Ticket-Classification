import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def upload_file(file_path):
    """Upload a file to OpenAI for fine-tuning."""
    print(f"Uploading {file_path}...")
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose="fine-tune"
            )
        print(f"File uploaded successfully. File ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return None

def create_fine_tuning_job(training_file_id):
    """Create a fine-tuning job."""
    print("Creating fine-tuning job...")
    try:
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model="gpt-3.5-turbo-0125"  # Using the latest version of GPT-3.5-turbo
        )
        print(f"Fine-tuning job created successfully. Job ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {str(e)}")
        return None

def monitor_fine_tuning_job(job_id):
    """Monitor the progress of a fine-tuning job."""
    print("Monitoring fine-tuning progress...")
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            print(f"Current status: {status}")
            
            if status == "succeeded":
                print("Fine-tuning completed successfully!")
                return job.fine_tuned_model
            elif status == "failed":
                print("Fine-tuning failed!")
                if hasattr(job, 'error'):
                    print(f"Error details: {job.error}")
                return None
            elif status == "validating_files":
                print("Validating training files...")
                time.sleep(10)  # Check more frequently during validation
            elif status in ["running", "queued"]:
                print(f"Status: {status}")
                time.sleep(60)  # Check every minute
            else:
                print(f"Unexpected status: {status}")
                if hasattr(job, 'error'):
                    print(f"Error details: {job.error}")
                return None
        except Exception as e:
            print(f"Error monitoring job: {str(e)}")
            return None

def main():
    # Check if training data exists
    training_file = Path("data/training_data.jsonl")
    if not training_file.exists():
        print("Training data not found. Please run generate_data.py first.")
        return

    # Upload training file
    training_file_id = upload_file(training_file)
    if not training_file_id:
        print("Failed to upload training file.")
        return
    
    # Create fine-tuning job
    job_id = create_fine_tuning_job(training_file_id)
    if not job_id:
        print("Failed to create fine-tuning job.")
        return
    
    # Monitor the job
    fine_tuned_model = monitor_fine_tuning_job(job_id)
    
    if fine_tuned_model:
        print(f"Fine-tuned model: {fine_tuned_model}")
        # Save the model name for later use
        with open("data/fine_tuned_model.txt", "w") as f:
            f.write(fine_tuned_model)
    else:
        print("Fine-tuning did not complete successfully.")

if __name__ == "__main__":
    main() 