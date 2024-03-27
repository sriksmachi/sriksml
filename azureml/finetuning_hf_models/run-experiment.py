from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    VsCodeJobService,
    TensorBoardJobService,
    JupyterLabJobService,
)


# Function to create MLClient
def create_ml_client(credential=None):     
    if credential is None:
        try:
            credential = DefaultAzureCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            credential = AzureCliCredential()

    return MLClient.from_config(credential=credential)

# Function to create AML Compute Cluster
def create_aml_cluster(ml_client, compute_cluster_name = "AmlComputeCluster", vm_size = "Standard_NC4as_T4_v3", min_nodes = 1, max_nodes = 2):
    # If you already have a gpu cluster, mention it here. Else will create a new one    
    try:
        compute = ml_client.compute.get(compute_cluster_name)
        print("successfully fetched compute:", compute.name)
    except Exception as ex:
        print("failed to fetch compute:", compute_cluster_name)
        print(f"creating new {vm_size} compute")
        compute = AmlCompute(
            name=compute_cluster_name,
            size=vm_size,
            min_instances=min_nodes,
            max_instances=max_nodes,  # For multi node training set this to an integer value more than 1
        )
        ml_client.compute.begin_create_or_update(compute).wait()
        print("successfully created compute:", compute.name)
    return compute

# Function to create Environment
def create_environment(ml_client, env_Name = "finetune_hf_lora"):
    try:
        env = ml_client.environments.get(env_Name)
        print("successfully fetched environment:", env.name)
    except Exception as ex:
        print("failed to fetch environment:", env_Name)
        print(f"creating new environment {env_Name}")
        env_docker_context = Environment(
            build=BuildContext(path="env"),
            name=env_Name,
            description="Environment created from a Docker context.",
        )
        ml_client.environments.create_or_update(env_docker_context)
        print("successfully created environment:", env_docker_context.name)
    return env_docker_context

# Function to create Job
def create_job(compute_cluster, script_file = "finetune_hf_models.py", job_name = "hf_finetuning", env_name = "finetune_hf_lora"):
    job = command(
        code=".",
        command=f"python {script_file} \
            --model_name google/flan-t5-small \
            --dataset squad \
            --num_epochs 1 \
            --target_input_length=512 \
            --target_max_length=100 \
            --train_size=1000",            
    compute=compute_cluster,
    services={
      "My_jupyterlab": JupyterLabJobService(
        nodes="all" # For distributed jobs, use the `nodes` property to pick which node you want to enable interactive services on. If `nodes` are not selected, by default, interactive applications are only enabled on the head node. Values are "all", or compute node index (for ex. "0", "1" etc.)
      ),
      "My_vscode": VsCodeJobService(
        nodes="all"
      ),
      "My_tensorboard": TensorBoardJobService(
        nodes="all",
        log_dir="outputs/runs"  # relative path of Tensorboard logs (same as in your training script)         
      ),
    },
    environment=f"{env_name}@latest",
    instance_count=1,  
    display_name=job_name)
    return job
 
if __name__ == "__main__":
    compute_cluster_name = "AmlComputeCluster"
    ml_client = create_ml_client()
    compute_cluster = create_aml_cluster(ml_client, compute_cluster_name = compute_cluster_name)
    env_docker_context = create_environment(ml_client)
    job = create_job(compute_cluster_name)
    print(f"Creating job : {job}")
    job = ml_client.jobs.create_or_update(job)
    print("Job created successfully")
    print(job.studio_url)

    