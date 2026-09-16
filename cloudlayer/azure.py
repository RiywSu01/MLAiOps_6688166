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

import os
import time
import subprocess

from pathlib import Path
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, AzureBlobDatastore, AccountKeyConfiguration


class AzureAdapter(CloudAdapter):
    @staticmethod
    def _parse_blob_uri(blob_uri: str) -> tuple[str, str, str]:
        """Parse BLOB_URI into (account_url, container, prefix).
        
        Supports both forms:
          - abfss://container@account.dfs.core.windows.net/prefix
          - https://account.blob.core.windows.net/container/prefix
        """
        parsed = urlparse(blob_uri)
        if parsed.scheme == "abfss":
            netloc_parts = parsed.netloc.split("@")
            if len(netloc_parts) == 2:
                container = netloc_parts[0]
                host = netloc_parts[1]
            else:
                container = ""
                host = parsed.netloc
            account_name = host.split(".")[0]
            account_url = f"https://{account_name}.blob.core.windows.net"
            prefix = parsed.path.strip("/")
            return account_url, container, prefix
        elif parsed.scheme in ("https", "http"):
            account_url = f"{parsed.scheme}://{parsed.netloc}"
            path = parsed.path.strip("/")
            if not path:
                raise ValueError(f"Invalid BLOB_URI: {blob_uri}")
            parts = path.split("/", 1)
            container = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            return account_url, container, prefix
        else:
            raise ValueError(f"Unsupported BLOB_URI scheme: {parsed.scheme}")

    def upload(self, local_path: str, key: str) -> str:
        #Get the url, container, prefix from _parse_blob_uri function
        #use "self.cfg" because the base.py already declared
        account_url, container, prefix = self._parse_blob_uri(self.cfg.blob_uri)

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
        account_url, container, blob_name = self._parse_blob_uri(uri)

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
        digests = [d.strip() for d in result.stdout.strip().splitlines() if d.startswith(registry + "@")]
        if not digests:
            raise RuntimeError(
                f"Could not find digest for {remote_tag}"
            )
        return digests[0]
    
    def _get_ml_client(self) -> MLClient:
        # Helper to instantiate MLClient connected to the Azure ML Workspace.
        # Get subscription ID from env or via az CLI
        sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        if not sub_id:
            try:
                sub_id = subprocess.check_output(
                    ["az", "account", "show", "--query", "id", "-o", "tsv"],
                    text=True,
                ).strip()
            except Exception:
                sub_id = None
        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=sub_id,
            resource_group_name=self.cfg.project_id,
            workspace_name=self.cfg.project_id,
        )

    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        """Submit a containerised training job to Azure ML serverless compute.
        
        Takes:
          - image_uri: ACR reference (e.g. registry.azurecr.io/itcs355:67edae4)
          - args: training arguments dict (e.g. {"seed": 20260101, "n_estimators": 200})
        Returns:
          - job_id: unique Azure ML job name
        """
        ml_client = self._get_ml_client()

        account_url, container, prefix = self._parse_blob_uri(self.cfg.blob_uri)
        account_name = urlparse(account_url).hostname.split(".")[0]
        blob_key = f"{prefix}/raw/sensors.csv" if prefix else "raw/sensors.csv"

        # Ensure raw dataset exists in blob storage (Task 1 requirement: read from BLOB_URI)
        local_raw = Path(self.cfg.raw_path)
        if local_raw.exists():
            self.upload(str(local_raw), "raw/sensors.csv")
        else:
            # Check if dataset already exists in blob storage
            blob_service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob_key)
            if not blob_client.exists():
                raise FileNotFoundError(
                    f"sensors.csv not found locally at {local_raw} and not found in storage at {blob_key}. "
                    "Run `make data` first."
                )

        # Ensure Azure ML datastore exists for seamless compute data download
        try:
            ml_client.datastores.get(container)
        except Exception:
            account_key = subprocess.check_output(
                ["az", "storage", "account", "keys", "list", "--account-name", account_name, "--query", "[0].value", "-o", "tsv"],
                text=True,
            ).strip()
            ds = AzureBlobDatastore(
                name=container,
                account_name=account_name,
                container_name=container,
                credentials=AccountKeyConfiguration(account_key=account_key),
            )
            ml_client.datastores.create_or_update(ds)

        # Format args dictionary into CLI flags: {"n_estimators": 200} -> "--n-estimators 200"
        cli_args = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in args.items())

        # Azure ML automatically injects an azureml:// tracking URI into the job environment, which
        # crashes standard provider-neutral MLflow. Force our configured tracking URI or SQLite store.
        tracking_uri = self.cfg.mlflow_tracking_uri
        if tracking_uri.startswith("azureml://") or tracking_uri == "sqlite:///mlflow.db":
            tracking_uri = "sqlite:////app/mlflow.db"

        cmd = (
            f"mkdir -p /app/data/raw && "
            f"cp ${{{{inputs.data}}}} /app/data/raw/sensors.csv && "
            f"cd /app && "
            f"unset MLFLOW_RUN_ID MLFLOW_EXPERIMENT_ID MLFLOW_EXPERIMENT_NAME && "
            f"MLFLOW_TRACKING_URI='{tracking_uri}' python -m src.train {cli_args}"
        ).strip()

        # Azure ML Command Job
        datastore_path = (
            f"azureml://datastores/{container}/paths/{prefix}/raw/sensors.csv"
            if prefix else f"azureml://datastores/{container}/paths/raw/sensors.csv"
        )
        job = command(
            command=cmd,
            inputs={
                "data": Input(
                    type="uri_file",
                    path=datastore_path,
                    mode="download",
                )
            },
            environment=Environment(image=image_uri),
            instance_type="Standard_DS3_v2",  # From src/costs.py
            environment_variables={
                "BLOB_URI": self.cfg.blob_uri,
                "MLFLOW_TRACKING_URI": self.cfg.mlflow_tracking_uri,
            },
            tags=self.cfg.tags(2),             # Tag with course=itcs355, lab=2 for cost/teardown
            display_name=f"itcs355-lab2-{self.cfg.project_id}",
        )
        submitted_job = ml_client.jobs.create_or_update(job)
        return submitted_job.name
    
    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Poll Azure ML until the job reaches a terminal state (Finished state)."""
        ml_client = self._get_ml_client()
        print(f"Waiting for Azure ML job '{job_id}' to complete...")
        while True:
            job = ml_client.jobs.get(job_id)
            status = getattr(job.status, "value", str(job.status))
            print(f"  Job {job_id} status: {status}")
            if status in ("Completed", "Failed", "Canceled"):
                break
            time.sleep(15)
        if status != "Completed":
            raise RuntimeError(f"Azure ML training job {job_id} finished with status: {status}")
        studio_service = getattr(job, "services", {}).get("Studio") if hasattr(job, "services") and job.services else None
        studio_url = getattr(studio_service, "endpoint", "") if studio_service else ""
        return {
            "job_id": job.name,
            "status": status,
            "studio_url": studio_url,
        }

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
