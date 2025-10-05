import os
import boto3
from urllib.parse import urlparse

def download_model_from_s3(s3_uri: str, destination_path: str) -> None:
    """Завантажує файл моделі з S3 або MinIO до вказаного шляху"""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name=os.getenv("S3_REGION")
    )

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    s3_client.download_file(bucket, key, destination_path)
