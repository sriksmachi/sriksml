from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """    
    Model arguments for the pre-training model
    """
    model_name: str = field(
        metadata={"help": "Name of the pre-training language model from huggingface/models"},
        default="google/flan-t5-small"       
    )
    