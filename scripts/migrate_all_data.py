import os
import subprocess
from dotenv import load_dotenv

def extract_from_git(ref, output_path):
    print(f"Extracting {ref}...")
    result = subprocess.run(["git", "show", ref], capture_output=True)
    if result.returncode == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(result.stdout)
        print(f"Saved to {output_path}")
        return True
    else:
        print(f"Failed to extract {ref}: {result.stderr.decode('utf-8')}")
        return False

def main():
    load_dotenv()
    
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not repo_owner or not repo_name:
        print("ERROR: DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME missing from .env")
        return

    print("--- 1. Extracting historical data from Git ---")
    
    # 1. LLM JSONs (from commit c690a08 before deletion)
    json1 = "data/raw_jsons/scam_call_hard_examples_250_fable.json"
    json2 = "data/raw_jsons/scam_call_transcripts_250_combined_gpt5.6.json"
    extract_from_git("c690a08:data/synthesized_data/scam_call_hard_examples_250_fable.json", json1)
    extract_from_git("c690a08:data/synthesized_data/scam_call_transcripts_250_combined_gpt5.6.json", json2)
    
    # 2. Legacy Composite (from model-distilbert branch)
    csv1 = "data/legacy_composite/composite_train.csv"
    csv2 = "data/legacy_composite/composite_test.csv"
    extract_from_git("model-distilbert:data/processed/composite_train.csv", csv1)
    extract_from_git("model-distilbert:data/processed/composite_test.csv", csv2)

    print("\n--- 2. Uploading to DagsHub ---")
    try:
        from dagshub.upload import Repo
        repo = Repo(repo_owner, repo_name)
        
        for file in [json1, json2, csv1, csv2]:
            if os.path.exists(file):
                print(f"Uploading {file}...")
                repo.upload(file=file, path=file, commit_message=f"Archive {file}")
                
        print("Upload complete! All historical datasets are now centralized in DagsHub.")
    except Exception as e:
        print(f"Failed to upload to DagsHub: {e}")

if __name__ == "__main__":
    main()
