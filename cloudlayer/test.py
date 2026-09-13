from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


blob_uri = "https://mystorage.blob.core.windows.net/ml-data/project1"


def _parse_blob_uri(blob_uri):
    parsed = urlparse(blob_uri)

    url = parsed.scheme + "://" + parsed.netloc

    split_array = parsed.path.split("/")
    container = split_array[1]
    prefix = split_array[2]

    return url, container, prefix


url, container, prefix = _parse_blob_uri(blob_uri)

print("URL:", url)
print("Container:", container)
print("Prefix:", prefix)


credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=url,
    credential=credential
)

print("BlobServiceClient created!")