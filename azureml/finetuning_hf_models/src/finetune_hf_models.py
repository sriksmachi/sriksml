from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    set_seed,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
    )

from dataclasses import dataclass, field
from datasets import load_dataset
import torch
import nltk
import evaluate
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
import logging
import os
from entities.training_arguments import TrainingArguments
from entities.model_arguments import ModelArguments
import mlflow
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
# Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()        
    parser.add_argument("--tensorboard_log_dir", default="/outputs/tblogs/")    
    logging.basicConfig(level=training_args.log_level)    
    set_seed(training_args.seed)
    logger.info(f"Training arguments: {training_args}")
    
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

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        result = rogue_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        mlflow.log_metrics(result)
        return result

    def get_dataset(training_args):
        print(f"loading data from : {training_args.data_path}")
        squad = load_dataset('json', data_files=training_args.data_path, field="data", split='train')
        squad = squad.train_test_split(test_size=0.2)
        print(squad.shape)
        return squad

    def load_tokenizer_model(model_args):
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name, torch_dtype=torch.bfloat16)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name)
        return tokenizer, model, data_collator

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

    def load_metrics():
        # Load metrics
        try:
            nltk.download('punkt')
        except:
            logger.info("error downloading punkt")
        rogue_metric = evaluate.load('rouge')
        return rogue_metric

    def preprocess_dataset(squad, training_args):
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
    
    # Load the dataset, tokenizer, model, PEFT model, metrics and training arguments
    mlflow.set_tracking_uri(training_args.mlflow_tracking_uri)
    squad = get_dataset(training_args)
    tokenizer, model, data_collator = load_tokenizer_model(model_args)
    peft_model = load_peft_model(model, training_args)
    rogue_metric = load_metrics()
    tensored_data = preprocess_dataset(squad, training_args)
    seq_2_seq_training_args = load_training_arguments(training_args)

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=seq_2_seq_training_args,
        train_dataset=tensored_data["train"],
        eval_dataset=tensored_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # train the model    
    mlflow.create_experiment(training_args.experiment_name)
    experiment_id = mlflow.get_experiment_by_name(training_args.experiment_name).experiment_id
    mlflow.autolog()
    run_id = None
    artifact_path = "model"
    model_name = f"{training_args.experiment_name}_model"
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
    main()