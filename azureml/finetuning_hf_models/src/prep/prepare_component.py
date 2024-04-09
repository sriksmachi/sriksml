from datasets import load_dataset
import os
from pathlib import Path
from mldesigner import command_component, Input, Output
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    set_seed,
    HfArgumentParser,
    DataCollatorForSeq2Seq
    )
from datasets import load_dataset
import torch
import numpy as np
import logging
import os
from src.entities.training_arguments import TrainingArguments
from src.entities.model_arguments import ModelArguments
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
# Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_dataset(training_args):
        print(f"loading data from : {training_args.data_path}")
        data_files = {"train": training_args.data_path}
        squad = load_dataset('json', data_files=data_files, split='train')
        squad = squad.train_test_split(test_size=0.2)
        print(squad.shape)
        return squad

def load_tokenizer_model(model_args):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name, torch_dtype=torch.bfloat16)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name)
    return tokenizer, model, data_collator

def preprocess_dataset(squad, training_args, tokenizer):
        
        # Preprocess the data
        def preprocess_data(examples):
              """Adds prefix, tokenizes and sets the labels"""
              questions = examples["question"]
              contexts = examples["context"]
              titles = examples["title"]
              answers = []
              for answer in examples["answers.text"]:
                answers.append(answer[0])
              inputs = []
              for question, context in zip(questions, contexts):
                prefix = f"""Answer a question about this article in few sentences:\n{context}\nQ:{question}A:"""
                input = prefix.format(context=context.strip(), question=question.strip())
                inputs.append(input)
              model_inputs = tokenizer(inputs,
                                       truncation='longest_first',
                                       padding="max_length",
                                       max_length=training_args.target_input_length,
                                       return_tensors="pt")
              model_inputs["query"] = tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True)
              labels = tokenizer(text_target=answers, max_length=training_args.target_max_length, truncation=True)
              model_inputs["labels"] = labels["input_ids"]
              return model_inputs

        # Prepare data
        squad = squad.flatten()    
        filtered_squad = squad.filter(lambda x: (len(x.get('context')) + len(x.get('question')) < training_args.target_input_length) 
                                      and (x.get('answers.answer_start')[0]) < (training_args.target_input_length
                                                                                + len(x.get('answers.text'))) and len(x.get('answers.text')) > 0)
        filtered_squad = filtered_squad.shuffle()
        filtered_squad['train'] = filtered_squad['train'].select(range(training_args.train_size))
        filtered_squad['test'] = filtered_squad['test'].select(range(training_args.test_size))
        tensored_data = filtered_squad.map(preprocess_data, remove_columns=squad["train"].column_names, batched=True)
        tensored_data.set_format("pt", columns=["input_ids"], output_all_columns=True)
        return tensored_data

@command_component(
    name="prep_data",
    version="1",
    display_name="Prep Data",
    description="Convert data to tensored format"    
)
def prepare_data_component():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()        
    parser.add_argument("--tensorboard_log_dir", default="/outputs/tblogs/")    
    logging.basicConfig(level=training_args.log_level)    
    set_seed(training_args.seed)
    logger.info(f"Training arguments: {training_args}")
    dataset = get_dataset(training_args)
    tokenizer, _, _ = load_tokenizer_model(model_args)
    return preprocess_dataset(dataset, training_args, tokenizer)       