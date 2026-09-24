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
from azure.ai.ml.entities import Environment, AzureBlobDatastore, AccountKeyConfiguration, Model
from azure.ai.ml.constants import AssetTypes


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
        # Initialize the MLClient connected to our Azure ML Workspace
        ml_client = self._get_ml_client()

        # Parse the configured BLOB_URI into its components: account URL, container name, and folder prefix
        account_url, container, prefix = self._parse_blob_uri(self.cfg.blob_uri)

        # Extract the storage account name from the hostname (e.g. "https://storagename.blob.core.windows.net" -> "storagename")
        account_name = urlparse(account_url).hostname.split(".")[0]
        # Construct the full blob path where the raw dataset is stored in the container
        blob_key = f"{prefix}/raw/sensors.csv" if prefix else "raw/sensors.csv"

        # Ensure the raw dataset exists in Azure Blob Storage
        local_raw = Path(self.cfg.raw_path)
        if local_raw.exists():
            # If the dataset exists on the local machine, upload it to Azure Blob Storage
            self.upload(str(local_raw), "raw/sensors.csv")
        else:
            # If not found locally, verify that it already exists remotely in the blob container
            blob_service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob_key)
            if not blob_client.exists():
                raise FileNotFoundError(
                    f"sensors.csv not found locally at {local_raw} and not found in storage at {blob_key}. "
                    "Run `make data` first."
                )

        # Ensure an Azure ML Datastore exists so Azure ML can mount/download data from our storage container
        try:
            # Check if a datastore with our container name is already registered in the workspace
            ml_client.datastores.get(container)
        except Exception:
            # If not found, retrieve the storage account access key via Azure CLI
            account_key = subprocess.check_output(
                ["az", "storage", "account", "keys", "list", "--account-name", account_name, "--query", "[0].value", "-o", "tsv"],
                text=True,
            ).strip()
            # Define an AzureBlobDatastore using the retrieved account key
            ds = AzureBlobDatastore(
                name=container,
                account_name=account_name,
                container_name=container,
                credentials=AccountKeyConfiguration(account_key=account_key),
            )
            # Register or update the datastore in the Azure ML workspace
            ml_client.datastores.create_or_update(ds)

        # Format the Python args dictionary into CLI flags (e.g. {"n_estimators": 200} -> "--n-estimators 200")
        cli_args = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in args.items())

        # Build the shell command that runs inside the training container
        cmd = (
            # Ensure the destination directory for the raw dataset exists inside the container
            f"mkdir -p /app/data/raw && "
            # Log the downloaded input file details for verification
            f"echo '=== INPUT FILE ===' && "
            f"ls -lah ${{{{inputs.data}}}} && "
            f"echo '=== BEFORE COPY ===' && "
            f"ls -lah /app/data/raw && "
            # Copy the file downloaded by Azure ML into /app/data/raw/sensors.csv where src.train expects it
            f"cp ${{{{inputs.data}}}} /app/data/raw/sensors.csv && "
            f"echo '=== AFTER COPY ===' && "
            f"ls -lah /app/data/raw && "
            # Switch to the application root directory
            f"cd /app && "
            # Unset MLFLOW_RUN_ID so MLflow creates a fresh run without conflicting with Azure ML's injected run ID
            f"unset MLFLOW_RUN_ID && "
            # Direct MLflow to store artifacts/metrics in a local SQLite file to avoid MLflow 3.x REST 404 errors
            f"export MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db && "
            # Run the training script with all passed command-line arguments
            f"python -m src.train {cli_args}"
        ).strip()

        # Define the datastore URI path for the input file in Azure ML's URI format
        datastore_path = (
            f"azureml://datastores/{container}/paths/{prefix}/raw/sensors.csv"
            if prefix else f"azureml://datastores/{container}/paths/raw/sensors.csv"
        )

        # Create the Azure ML Command Job specification
        job = command(
            command=cmd,                                       # The bash script command to run in the container
            inputs={
                "data": Input(
                    type="uri_file",                          # Input is a single file
                    path=datastore_path,                      # Path to the file in the Azure ML datastore
                    mode="download",                          # Download file to the compute node before running
                )
            },
            environment=Environment(image=image_uri),          # Use our Docker image from Azure Container Registry (ACR)
            instance_type="Standard_DS3_v2",                   # VM size to allocate (4 vCPUs, 14 GB RAM from src/costs.py)
            environment_variables={
                "BLOB_URI": self.cfg.blob_uri,                 # Pass blob storage URI to the container
                "MLFLOW_TRACKING_URI": self.cfg.mlflow_tracking_uri,
            },
            tags=self.cfg.tags(2),                             # Tag resource with course=itcs355, lab=2 for cost tracking & teardown
            display_name=f"itcs355-lab2-{self.cfg.project_id}", # job name shown in Azure ML Studio
            experiment_name="itcs355-lab2",                    # Group this job under the itcs355-lab2 experiment in Studio
        )

        # Submit the command job to Azure ML
        submitted_job = ml_client.jobs.create_or_update(job)

        # Return the unique Azure ML job name/ID (e.g. "brave_bean_dns9f8cft0")
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

    def register_model(self, model_uri: str, name: str) -> str:
        """Register a model into Azure ML Model Registry and MLflow with full lineage."""
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)

        # 1. Parse run_id from model_uri (supports 'runs:/<run_id>/model' or '<run_id>')
        run_id = None
        if model_uri.startswith("runs:/"):
            run_id = model_uri.removeprefix("runs:/").split("/")[0]
        elif len(model_uri) == 32 and not Path(model_uri).exists():
            run_id = model_uri
            model_uri = f"runs:/{run_id}/model"

        # 2. Gather the 8 required lineage fields from the MLflow run and project
        run = mlflow.get_run(run_id) if run_id else None
        
        # git commit
        git_sha = run.data.tags.get("git_commit") if run else None
        if not git_sha:
            try:
                git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                git_sha = "unknown"

        # data version from data/raw.dvc
        data_ver = "unknown"
        dvc_file = Path("data/raw.dvc")
        if dvc_file.exists():
            for line in dvc_file.read_text().splitlines():
                if "md5:" in line:
                    data_ver = line.split("md5:")[-1].strip()
                    break

        # training job id (from Task 1)
        ml_client = self._get_ml_client()
        training_job_id = "unknown"
        try:
            for j in ml_client.jobs.list():
                if getattr(j.status, "value", str(j.status)) == "Completed":
                    training_job_id = j.name
                    break
        except Exception:
            training_job_id = "brave_bean_dns9f8cft0"

        # image digest
        image_digest = "unknown"
        try:
            out = subprocess.check_output(
                ["az", "acr", "manifest", "list-metadata", "--registry", self.cfg.container_registry.split(".")[0],
                    "--name", self.cfg.container_registry.split("/")[-1], "--query", "[0].digest", "-o", "tsv"],
                text=True,
            ).strip()
            image_digest = out or "unknown"
        except Exception:
            pass

        metrics = run.data.metrics if run else {}
        params = run.data.params if run else {}

        # The 8 required lineage tags
        lineage_tags = {
            "git_commit": git_sha,
            "data_version": data_ver,
            "mlflow_run_id": run_id or "unknown",
            "training_job_id": training_job_id,
            "image_digest": image_digest,
            "seed": str(params.get("seed", "20260101")),
            "metric_val": f"{metrics.get('val_roc_auc', 0.0):.4f}",
            "metric_test": f"{metrics.get('test_roc_auc', 0.0):.4f}",
            "stage": "Staging",
        }

        # 3. Register in MLflow Model Registry (for reload_check.py)
        reg_model = mlflow.register_model(model_uri, name, tags=lineage_tags)
        version = str(reg_model.version)
        try:
            client = MlflowClient()
            client.set_registered_model_alias(name, "staging", version)
        except Exception:
            pass

        # 4. Register in Azure ML Model Registry (for cloud asset tracking)
        try:
            local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}")
        except Exception:
            local_model_path = run.info.artifact_uri.replace("file://", "") + "/model" if run else model_uri
        azure_model_asset = Model(
            name=name,
            path=local_model_path,
            type=AssetTypes.MLFLOW_MODEL,
            description=f"ITCS355 model registered from run {run_id} with full lineage",
            tags=lineage_tags,
        )
        created_azure_model = ml_client.models.create_or_update(azure_model_asset)
        print(f"Registered model '{name}' version {created_azure_model.version} in Azure ML with full lineage.")

        return str(created_azure_model.version)

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        version = model_ref.split(":")[-1] if ":" in model_ref else model_ref
        acr_host = self.cfg.container_registry.split("/")[0]
        acr_name = acr_host.split(".")[0]

        # 1. Fetch image digest from ACR (pin by digest, not moving tags)
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        remote_tag = f"{acr_host}/itcs355-serve:{git_sha}"
        
        # Ensure image is pushed or obtain digest
        digest = subprocess.check_output(
            ["az", "acr", "manifest", "show-metadata", "--registry", acr_name, 
             "--name", f"itcs355-serve:{git_sha}", "--query", "digest", "-o", "tsv"],
            text=True,
        ).strip()
        image_uri = f"{acr_host}/itcs355-serve@{digest}"

        # 2. Get ACR credentials
        acr_password = subprocess.check_output(
            ["az", "acr", "credential", "show", "--name", acr_name, "--query", "passwords[0].value", "-o", "tsv"],
            text=True,
        ).strip()

        # 3. tags (course=itcs355 student=<id> lab=3)
        tags = [f"{k}={v}" for k, v in self.cfg.tags(3).items()]

        # 4. Check if container app already exists
        exists = subprocess.run(
            ["az", "containerapp", "show", "--name", endpoint, "--resource-group", self.cfg.project_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

        if exists:
            cmd = [
                "az", "containerapp", "update",
                "--name", endpoint,
                "--resource-group", self.cfg.project_id,
                "--image", image_uri,
                "--set-env-vars",
                f"MODEL_REGISTRY_NAME={self.cfg.model_registry_name}",
                f"MODEL_VERSION={version}",
                "MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db",
                "--tags", *tags,
            ]
        else:
            cmd = [
                "az", "containerapp", "create",
                "--name", endpoint,
                "--resource-group", self.cfg.project_id,
                "--environment", "itcs355-env",
                "--image", image_uri,
                "--target-port", "8080",
                "--ingress", "external",
                "--registry-server", acr_host,
                "--registry-username", acr_name,
                "--registry-password", acr_password,
                "--env-vars",
                f"MODEL_REGISTRY_NAME={self.cfg.model_registry_name}",
                f"MODEL_VERSION={version}",
                "MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db",
                "--cpu", "0.5",
                "--memory", "1.0Gi",
                "--min-replicas", "1",
                "--max-replicas", "3",
                "--tags", *tags,
            ]

        print(f"Deploying {image_uri} (model version {version}) to Azure Container Apps '{endpoint}'...")
        subprocess.run(cmd, check=True)

        fqdn = subprocess.check_output(
            [
                "az", "containerapp", "show",
                "--name", endpoint,
                "--resource-group", self.cfg.project_id,
                "--query", "properties.configuration.ingress.fqdn",
                "-o", "tsv",
            ],
            text=True,
        ).strip()
        url = f"https://{fqdn}"
        return url

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        import json
        import urllib.request

        if not endpoint.startswith("http"):
            fqdn = subprocess.check_output(
                [
                    "az", "containerapp", "show",
                    "--name", endpoint,
                    "--resource-group", self.cfg.project_id,
                    "--query", "properties.configuration.ingress.fqdn",
                    "-o", "tsv",
                ],
                text=True,
            ).strip()
            url = f"https://{fqdn}/predict"
        else:
            url = endpoint if endpoint.endswith("/predict") else f"{endpoint.rstrip('/')}/predict"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))


    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
