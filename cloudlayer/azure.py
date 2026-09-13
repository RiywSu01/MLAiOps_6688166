"""Azure adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install azure-storage-blob azure-identity azure-containerregistry
Docs: BlobServiceClient for storage; ACR push goes through `docker push` after
      `az acr login --name <registry>`.

Hints for Lab 1:
  * BLOB_URI is either abfss://container@account.dfs.core.windows.net/prefix or
    https://account.blob.core.windows.net/container/prefix. Pick one form and parse
    it here, never in src/.
  * Use DefaultAzureCredential rather than a connection string. It picks up your CLI
    login locally and your managed identity in CI, which is what Lab 4 needs.
  * push_image must return the digest reference: registry.azurecr.io/repo@sha256:...
  * Azure tags live on the resource, not the blob. Tag the storage account, the
    registry, and later the workspace with cfg.tags(1).
"""
from __future__ import annotations

from typing import Any

import subprocess

from pathlib import Path
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


class AzureAdapter(CloudAdapter):
    #Parse blob-uri function
    def _parse_blob_uri(blob_uri: str) -> tuple[str, str, str]:
        parsed = urlparse(blob_uri)

        account_url = f"{parsed.scheme}://{parsed.netloc}"

        parts = parsed.path.strip("/").split("/", 1)

        if not parts:
            raise ValueError(f"Invalid BLOB_URI: {blob_uri}")

        container = parts[0]
        #Check that second element exist or not on prefix
        prefix = parts[1] if len(parts) > 1 else ""

        return account_url, container, prefix;

    def upload(self, local_path: str, key: str) -> str:
        #Get the url, container, prefix from _parse_blob_uri function
        #use "self.cfg" because the base.py already declared
        account_url, container, prefix = _parse_blob_uri(self.cfg.blob_uri)

        #Azure figure out how this app should authenticate by itself, because we dont want to used connection string due to it can leaks the crucial information.
        credential = DefaultAzureCredential()

        # Like create the client representing my Azure Storage Account.
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )
   
        #Get container with the container name we extract from the _parse_blob_uri function
        container_client = blob_service_client.get_container_client(container)

        #blob_name is like a filepath of the azure storage
        blob_name = f"{prefix}/{key}" if prefix else key

        #Get the actual blob client name
        blob_client = container_client.get_blob_client(blob_name)

        #Open local file
        # "rb" means read the file as binary data because we dont know file format so we treat it as bytes.
        with open(local_path, "rb") as data:
            #upload local file to Azure blob Storage
            blob_client.upload_blob(data, overwrite=True)

        return blob_client.url


    def download(self, uri: str, local_path: str) -> None:
        account_url, container, blob_name = _parse_blob_url(uri)

        credential = DefaultAzureCredential()

        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )

        container_client = blob_service_client.get_container_client(container)

        blob_client = container_client.get_blob_client(blob_name)

        #check if the local_path exist on local disk, if missing automatically create any missing folder.
        Path(local_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        #.readinto(file_handle): Streams the data directly into a local file chunk-by-chunk, avoiding high memory usage for large files.
        
        with open(local_path, "wb") as file:
            blob_client.download_blob().readinto(file)

  
    def push_image(self, local_tag: str) -> str:
        #Get container_registry values from env
        registry = self.cfg.container_registry

        #Get Azure registry resource name
        #Remove .azurecr.io out of the registry name
        server = registry.split("/")[0]   # "Testname.azurecr.io"
        registry_name = server.split(".")[0] # "Tesname"

        #Extract git tag like (:30ba98b) and form remote tag = Testname.azurecr.io/itcs355:30ba98b
        tag = local_tag.split(":")[-1]
        remote_tag = f"{registry}:{tag}"

        #Login to ACR, form a command shell structure ("az acr login --name myregistry")
        subprocess.run(
            ["az", "acr", "login", "--name", registry_name],
            check=True,
        )

        #Tag the local image with the remote name
        subprocess.run(
            ["docker", "tag", local_tag, remote_tag],
            check=True,
        )

        #Push the image
        subprocess.run(
            ["docker", "push", remote_tag],
            check=True,
        )

        #Make the docker return immutable digest reference (@sha256:...)
        result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                remote_tag,
                "--format",
                "{{range .RepoDigests}}{{println .}}{{end}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        #Make it output in this format ("registry.azurecr.io/repo@sha256:...")
        digests = [d.strip() for d in result.stdout.strip().splitlines() if registry_name in d]
        digest_reference = digests[0] if digests else result.stdout.strip().splitlines()[0]
        return digest_reference

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
