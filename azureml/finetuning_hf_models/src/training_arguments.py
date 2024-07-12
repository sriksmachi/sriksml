from dataclasses import dataclass, field

@dataclass
class TrainingArguments:  
    """
    Training arguments for the model
    """
    data_path: str = field(
        metadata={"help": "Path to datafile to use for training"},
        default="./data/squad.json"
    )
    seed: int = field(
        metadata={"help": "Seed for the model"},
        default=42
    )
    target_input_length: int = field(
        metadata={"help": "Maximum length of the input text"},
        default=512
    )
    target_max_length: int = field(
        metadata={"help": "Maximum length of the target text"},
        default=100
    )
    train_size: int = field(
        metadata={"help": "Number of training examples"},
        default=1000
    )
    test_size: int = field(
        metadata={"help": "Number of test examples"},
        default=300
    )
    learning_rate: int = field(
        metadata={"help": "Learning rate for the model"},
        default=1e-3
    )
    batch_size: int = field(
        metadata={"help": "Batch size for training"},
        default=8
    )
    per_device_eval_batch: int = field(
        metadata={"help": "Batch size for evaluation"},
        default=8
    )
    weight_decay: float = field(
        metadata={"help": "Weight decay for the model"},
        default=0.01
    )
    save_total_lim: int = field(
        metadata={"help": "Number of checkpoints to save"},
        default=3
    )
    num_epochs: int = field(
        metadata={"help": "Number of epochs to train the model"},
        default=1
    )
    lora_rank: int = field(
        metadata={"help": "Rank of the LoRA matrix"},
        default=16
    )
    log_level: str = field(
        metadata={"help": "Logging level"},
        default="INFO"
    )
    lora_rank: int = field(
        metadata={"help": "Rank of the LoRA matrix"},
        default=16
    )
    lora_alpha: int = field(
        metadata={"help": "Alpha value for LoRA"},
        default=32
    )
    lora_dropout: float = field(
        metadata={"help": "Dropout value for LoRA"},
        default=0.05
    )
    mlflow_tracking_uri: str = field(
        metadata={"help": "MLFlow tracking URI"},
        default="http://localhost:8080"
    )
    experiment_name: str = field(
        metadata={"help": "Name of the experiment"},
        default="hf_finetuning"
    )
    output_dir: str = field(
        metadata={"help": "Output directory"},
        default="outputs"
    )
        
