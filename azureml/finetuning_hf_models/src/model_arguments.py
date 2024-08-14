from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """    
    Model arguments for the pre-training model
    """
    model_name: str = field(
        metadata={"help": "Name of the pre-training language model from huggingface/models"},
        default="microsoft/Phi-3-mini-4k-instruct"       
    )
    