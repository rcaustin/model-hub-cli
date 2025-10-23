from dotenv import load_dotenv
import os, boto3

load_dotenv()  # reads .env in repo root

region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
table_name = os.getenv("DYNAMO_TABLE_NAME")
bucket = os.getenv("S3_BUCKET")

dynamo = boto3.resource("dynamodb", region_name=region)
s3 = boto3.client("s3", region_name=region)

print("== ENV ==")
print("region:", region, "table:", table_name, "bucket:", bucket)

# DynamoDB table status
print("\n== DynamoDB ==")
print("status:", dynamo.Table(table_name).table_status)

# S3 tiny round-trip
print("\n== S3 round trip ==")
key = "healthcheck.txt"
s3.put_object(Bucket=bucket, Key=key, Body=b"ok")
obj = s3.get_object(Bucket=bucket, Key=key)
print("body:", obj["Body"].read().decode())
s3.delete_object(Bucket=bucket, Key=key)

print("\nALL GOOD ✅")
