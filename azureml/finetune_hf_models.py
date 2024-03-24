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

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """    
    Model arguments for the pre-training model
    """
    model_name: str = field(
        metadata={"help": "Name of the pre-training language model from huggingface/models"}
        default="google/flan-t5-small"       
    )
    
    
@dataclass
class TrainingArguments:  
    """
    Training arguments for the model
    """
    dataset: str = field(
        metadata={"help": "Dataset to use for training"},
        default="squad"
    )
    seed: int = field(
        metadata={"help": "Seed for the model"},
        default=42
    )
    target_input_length: str = field(
        metadata={"help": "Maximum length of the input text"},
        default=512
    )
    target_max_length: str = field(
        metadata={"help": "Maximum length of the target text"},
        default=100
    )
    train_size: str = field(
        metadata={"help": "Number of training examples"},
        default=1000
    )
    test_size: str = field(
        metadata={"help": "Number of test examples"},
        default=300
    )
    LEARNING_RATE: str = field(
        metadata={"help": "Learning rate for the model"},
        default=1e-3
    )
    BATCH_SIZE: str = field(
        metadata={"help": "Batch size for training"},
        default=8
    )
    PER_DEVICE_EVAL_BATCH: str = field(
        metadata={"help": "Batch size for evaluation"},
        default=8
    )
    WEIGHT_DECAY: str = field(
        metadata={"help": "Weight decay for the model"},
        default=0.01
    )
    SAVE_TOTAL_LIM: str = field(
        metadata={"help": "Number of checkpoints to save"},
        default=3
    )
    NUM_EPOCHS: str = field(
        metadata={"help": "Number of epochs to train the model"},
        default=5
    )
    lora_rank: str = field(
        metadata={"help": "Rank of the LoRA matrix"},
        default=16
    )
    
    
def main():
    
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    parser.add_argument("--tensorboard_log_dir", default="/outputs/tblogs/")
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)
    
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
      model_inputs = tokenizer(text=inputs,
                               max_length=training_args.target_max_length)
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
        return result

    # 1. Load the dataset
    squad = load_dataset(training_args.dataset, 
                         split="train")
    squad = squad.train_test_split(test_size=0.2)
    squad = squad.flatten()    
    filtered_squad = squad.filter(lambda x: (len(x.get('context')) + len(x.get('question')) < training_args.target_input_length) 
                                  and (x.get('answers.answer_start')[0]) < (training_args.target_input_length
                                                                            + len(x.get('answers.text'))) and len(x.get('answers.text')) > 0)
    filtered_squad = filtered_squad.shuffle()
    filtered_squad['train'] = filtered_squad['train'].select(range(training_args.train_size))
    filtered_squad['test'] = filtered_squad['test'].select(range(training_args.test_size))
    tensored_data = filtered_squad.map(preprocess_data, remove_columns=squad["train"].column_names, batched=True)
    tensored_data.set_format("pt", columns=["input_ids"], output_all_columns=True)
    
    # 2. Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name, torch_dtype=torch.bfloat16)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_args.model_name)
    
    # 3. Load the PEFT model    
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["lm_head"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM) 
    peft_model = get_peft_model(model, lora_config)
    peft_model = peft_model.to("cuda")    
    peft_model.print_trainable_parameters()
    
    # 4. Load metrics
    nltk.download('punkt', quiet=True)
    rogue_metric = evaluate.load('rouge')
    
    # 5. Load the training arguments
    training_args = Seq2SeqTrainingArguments(
    output_dir="./peft_results",
    learning_rate=training_args.LEARNING_RATE,
    num_train_epochs=training_args.NUM_EPOCHS,
    evaluation_strategy="epoch",
    predict_with_generate=True,
    per_device_train_batch_size=training_args.BATCH_SIZE,
    per_device_eval_batch_size=training_args.PER_DEVICE_EVAL_BATCH,
    weight_decay=training_args.WEIGHT_DECAY,
    save_total_limit=training_args.SAVE_TOTAL_LIM,
    push_to_hub=False
    )

    trainer = Seq2SeqTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=tensored_data["train"],
            eval_dataset=tensored_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
    )
    
    # 6. Train the model    
    trainer.train()
    
if __name__ == "__main__":
    main()