import os
import dagshub
from dotenv import load_dotenv

load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

dagshub.auth.add_app_token(token)
s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")

print("Listing models in DagsHub S3...")
response = s3_client.list_objects_v2(Bucket=repo_name, Prefix="models/")
if "Contents" in response:
    for obj in response["Contents"]:
        size_mb = obj["Size"] / (1024 * 1024)
        print(f" - {obj['Key']} ({size_mb:.2f} MB)")
else:
    print("No files found in models/ directory.")
