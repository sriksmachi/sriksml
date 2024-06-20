from pathlib import Path
from mldesigner import command_component, Input, Output
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    set_seed,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
    )
import torch
import nltk
import evaluate
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
import logging
import os
from training_arguments import TrainingArguments
from model_arguments import ModelArguments
import mlflow
import warnings
import datetime

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
# Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_tokenizer_model(model_args):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name, torch_dtype=torch.bfloat16)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name)
    return tokenizer, model, data_collator

def compute_metrics(eval_preds, tokenizer):
        def load_metrics():
            # Load metrics
            try:
                nltk.download('punkt')
            except:
                logger.info("error downloading punkt")
            rogue_metric = evaluate.load('rouge')
            return rogue_metric
    
        preds, labels = eval_preds
        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        rogue_metric = load_metrics()
        result = rogue_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        mlflow.log_metrics(result)
        return result

def load_training_arguments(training_args):
     # Load the training arguments
     training_args = Seq2SeqTrainingArguments( 
     output_dir="outputs",                                           
     learning_rate=training_args.learning_rate,
     num_train_epochs=training_args.num_epochs,
     evaluation_strategy="epoch",
     predict_with_generate=True,
     per_device_train_batch_size=training_args.batch_size,
     per_device_eval_batch_size=training_args.per_device_eval_batch,
     weight_decay=training_args.weight_decay,
     save_total_limit=training_args.save_total_lim,
     push_to_hub=False
     )
     return training_args
 
def load_peft_model(model, training_args):
        # Load the PEFT model
        rank = training_args.lora_rank 
        alpha = training_args.lora_alpha
        dropout = training_args.lora_dropout
        lora_config = LoraConfig(r=rank, lora_alpha=alpha, 
                                 target_modules=["lm_head"], lora_dropout=dropout, 
                                 bias="none", task_type=TaskType.SEQ_2_SEQ_LM) 
        logger.info(f"LoRA config: {lora_config}")
        peft_model = get_peft_model(model, lora_config)
        # peft_model = peft_model.to("cuda")    
        peft_model.print_trainable_parameters()
        return peft_model

def training_component(input_data: Input(type="uri_folder")):
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()        
    parser.add_argument("--tensorboard_log_dir", default="/outputs/tblogs/")    
    logging.basicConfig(level=training_args.log_level)    
    set_seed(training_args.seed)
    logger.info(f"Training arguments: {training_args}")
    mlflow.set_tracking_uri(training_args.mlflow_tracking_uri)
    tokenizer, model, data_collator = load_tokenizer_model(model_args)
    peft_model = load_peft_model(model, training_args)
    seq_2_seq_training_args = load_training_arguments(training_args)

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=seq_2_seq_training_args,
        train_dataset=input_data["train"],
        eval_dataset=input_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # train the model    
    experiment_name = f"{training_args.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    mlflow.create_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.autolog()
    run_id = None
    artifact_path = "model"
    model_name = f"{experiment_name}_model"
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_text(f"mlflow run_id: {run.info.run_id}", artifact_file=f"{run_id}.txt")
        mlflow.log_params(seq_2_seq_training_args.to_dict())
        trainer.train()
        trainer.evaluate()
        mlflow.transformers.log_model(
            transformers_model = {
                "model": peft_model,
                "tokenizer": tokenizer
            },
            artifact_path = artifact_path)   
    
    run = mlflow.get_run(run_id)
    print("run_id:", run.info.run_id)      
    mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
    
    
if __name__ == "__main__":
    training_component()