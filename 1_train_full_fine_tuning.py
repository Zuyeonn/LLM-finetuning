import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
#from trl.commands.cli_utils import  TrlParser -> 오류로 아래 코드 이용 혹은 pip install --force-reinstall -v "triton==3.1.0"
from trl.scripts.utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig
from sklearn.model_selection import train_test_split

import yaml
import requests

# Load dataset from the hub

from huggingface_hub import login

login(
    token="토큰 입력",
    add_to_git_credential=True
)

# 데이터셋 경로 기본값 설정
default_dataset_path = "C:/Users/user/juyeon/data"

dataset = load_dataset("beomi/KoAlpaca-v1.1a")
columns_to_remove = list(dataset["train"].features)

system_prompt = "당신은 다양한 분야의 전문가들이 제공한 지식과 정보를 바탕으로 만들어진 AI 어시스턴트입니다. 사용자들의 질문에 대해 정확하고 유용한 답변을 제공하는 것이 당신의 주요 목표입니다. ..."

train_dataset = dataset.map(
    lambda sample: {
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['instruction']},
            {"role": "assistant", "content": sample['output']}
        ]
    },
)

train_dataset = train_dataset.map(remove_columns=columns_to_remove, batched=False)
train_dataset = train_dataset["train"].train_test_split(test_size=0.1, seed=42)

# 데이터셋을 지정된 경로에 저장
os.makedirs(default_dataset_path, exist_ok=True)
train_dataset["train"].to_json(os.path.join(default_dataset_path, "train_dataset.json"), orient="records", force_ascii=False)
train_dataset["test"].to_json(os.path.join(default_dataset_path, "test_dataset.json"), orient="records", force_ascii=False)

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

# GitHub Raw 파일 URL
GITHUB_YAML_URL = "https://raw.githubusercontent.com/Zuyeonn/LLM-finetuning/refs/heads/main/0_full_fine_tuning_config.yaml"

# GitHub에서 YAML 파일 다운로드
response = requests.get(GITHUB_YAML_URL)

if response.status_code == 200:
    config = yaml.safe_load(response.text)  # YAML 파싱
else:
    raise ValueError("Failed to fetch YAML file from GitHub. Check the URL.")

@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=config.get("dataset_path", default_dataset_path),  # YAML에서 불러오기
        metadata={"help": "데이터셋 파일 경로"}
    )
    model_name: str = field(
        default=config.get("model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct"),  # YAML에서 불러오기
        metadata={"help": "SFT 학습에 사용할 모델 ID"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "SFT Trainer에 사용할 최대 시퀀스 길이"}
    )
    question_key: str = field(
        default=None, metadata={"help": "지시사항 데이터셋의 질문 키"}
    )
    answer_key: str = field(
        default=None, metadata={"help": "지시사항 데이터셋의 답변 키"}
    )

def training_function(script_args, training_args):
    # dataset_path가 None이거나 모델 경로로 지정되었으면 올바른 경로로 설정
    if script_args.dataset_path is None or "meta-llama" in script_args.dataset_path:
        script_args.dataset_path = default_dataset_path
    
    # 절대 경로 변환 (경로 문제 방지)
    train_dataset_path = os.path.abspath(os.path.join(script_args.dataset_path, "train_dataset.json"))
    test_dataset_path = os.path.abspath(os.path.join(script_args.dataset_path, "test_dataset.json"))

    print("Final Train Dataset Path:", train_dataset_path)
    print("Final Test Dataset Path:", test_dataset_path)
    
    # 파일 존재 여부 확인
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"Train dataset not found: {train_dataset_path}")
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")
    
    train_dataset = load_dataset("json", data_files=train_dataset_path, split="train")
    test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        attn_implementation="sdpa", 
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,  
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        },
    )
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    set_seed(training_args.seed)
    
    training_function(script_args, training_args)
