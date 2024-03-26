from mlora.modelargs import LoraConfig
from mlora.dispatcher import TrainTask, Dispatcher
from mlora.tasks import CasualTask, task_dict
from mlora.model import LLMModel

from transformers import get_scheduler
from typing import Dict, List, Union
import logging
import torch
import json
import os


class TrainConfig:
    def __init__(self,
                 train_config: Dict[str, any],
                 lora_config: LoraConfig):
        self.adapter_name_ = lora_config.adapter_name_
        self.batch_size_ = train_config["batch_size"]
        self.micro_batch_size_ = train_config["micro_batch_size"]
        self.optimizer_name_ = train_config["optim"]
        self.learning_rate_ = train_config["lr"]
        self.momentum_ = train_config.get("momentum", 0)
        self.weight_decay_ = train_config.get("weight_decay", 0.01)
        # Scheduler Types
        #   linear, cosine, cosine_with_restarts, polynomial, constant
        #   constant_with_warmup, inverse_sqrt, reduce_lr_on_plateau
        self.scheduler_type_: str = train_config.get(
            "scheduler_type", "linear")
        self.warmup_steps_: Union[int, float] = train_config.get(
            "warmup_steps", 0)
        self.all_training_steps_: int = -1
        self.lr_scheduler_: torch.optim.lr_scheduler.LRScheduler = None
        self.accumulation_step_: int = None
        self.accumulation_step_cnt_: int = 0
        self.optimizer_: torch.optim.Optimizer = None
        task_name = train_config.get("task_name", "casual")
        if task_name == "casual":
            self.task_ = CasualTask(
                data_path=train_config["data"],
                prompt_template=train_config.get("prompt", None),
                validation_size=train_config.get("val_set_size", None))
        else:
            self.task_ = task_dict[task_name]
        train_config["dataloader"] = self.task_.loading_data

    def prepare(self, train_paramas: List[torch.Tensor]):
        if self.batch_size_ < self.micro_batch_size_ or self.batch_size_ % self.micro_batch_size_ != 0:
            raise ValueError(
                f"error batch_size {self.batch_size_} and micro batch size {self.micro_batch_size_}")
        self.accumulation_step_ = self.batch_size_ / self.micro_batch_size_
        self.accumulation_step_cnt_ = 0
        paramas_count = sum(t.numel()
                            for t in train_paramas if t.requires_grad)
        logging.info(
            f"{self.adapter_name_} total trainable params: {paramas_count}")
        if self.optimizer_name_ == "sgd":
            self.optimizer_ = torch.optim.SGD(
                train_paramas, lr=self.learning_rate_,
                momentum=self.momentum_, weight_decay=self.weight_decay_)
        elif self.optimizer_name_ == "adamw":
            self.optimizer_ = torch.optim.AdamW(
                train_paramas, lr=self.learning_rate_, weight_decay=self.weight_decay_)
        else:
            raise ValueError(f"unkown optimizer {self.optimizer_name_}")

    def prepare_lr_scheduler(self, total_epoch, len_dataset):
        if self.lr_scheduler_ is None:
            total_steps = (len_dataset // self.batch_size_) * total_epoch if len_dataset % self.batch_size_ == 0 else (
                len_dataset // self.batch_size_ + 1) * total_epoch
            warmup_steps = self.warmup_steps_ * \
                total_steps if isinstance(
                    self.warmup_steps_, float) else self.warmup_steps_
            self.lr_scheduler_ = get_scheduler(
                self.scheduler_type_, self.optimizer_, warmup_steps, total_steps)

    def step(self):
        self.accumulation_step_cnt_ += 1
        if self.accumulation_step_cnt_ % self.accumulation_step_ == 0:
            self.optimizer_.step()
            self.lr_scheduler_.step()
            self.optimizer_.zero_grad()

    def finish(self):
        self.optimizer_.step()
        self.optimizer_.zero_grad()


def save_adapter_weight(model: LLMModel, config: TrainConfig, path: str, dir_suffix=""):
    lora_output_dir = path + os.sep + config.adapter_name_
    if dir_suffix != "":
        lora_output_dir += os.sep + \
            config.adapter_name_ + "_" + dir_suffix

    if not os.path.exists(lora_output_dir):
        os.makedirs(lora_output_dir)

    lora_weight_dict = model.get_lora_weight_dict(config.adapter_name_)
    lora_config_dict = model.adapter_configs_[config.adapter_name_].export()
    lora_config_dict["base_model_name_or_path"] = model.name_or_path_

    torch.save(lora_weight_dict, lora_output_dir +
               os.sep + "adapter_model.bin")

    with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
        json.dump(lora_config_dict, f, indent=4)


def train(dispatcher: Dispatcher,
          model: LLMModel,
          configs: List[TrainConfig],
          save_dir: str = ".",
          save_step: int = 2000) -> None:
    config_dict = {}
    for config in configs:
        config_dict[config.adapter_name_] = config

    def task_in_callback(task: TrainTask):
        adapter_name = task.adapter_name_
        logging.info(f"Loading training task {adapter_name}")
        config = config_dict[adapter_name]
        config.prepare(train_paramas[adapter_name])
        config.prepare_lr_scheduler(
            task.total_epoch_num_, len(task.train_token_data_))

    dispatcher.train_task_in_event_.register(task_in_callback)

    train_paramas = model.get_train_paramas()

    max_memory = 0

    step_cnt = 0
    while not dispatcher.check_task_done():
        labels, input = dispatcher.get_train_data()

        step_cnt += 1

        outputs = model.forward(input, labels)

        total_loss = None
        for output in outputs:
            adapter_name = output.adapter_name
            loss = output.loss / config_dict[adapter_name].accumulation_step_
            logging.info(
                f"    adapter: {adapter_name} loss: {loss}")
            if output.aux_loss:
                aux_loss = output.aux_loss / \
                    config_dict[adapter_name].accumulation_step_
                logging.info(
                    f"    adapter: {adapter_name}  aux: {aux_loss}")
                loss += aux_loss
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()

        for output in outputs:
            config = config_dict[output.adapter_name]
            config.step()

            max_memory = max(max_memory, torch.cuda.max_memory_allocated())
            logging.info(f"Max memory: {max_memory}")

            if config.accumulation_step_cnt_ % save_step == 0:
                save_adapter_weight(model, config, save_dir, f"{step_cnt}")

    for config in configs:
        config.finish()
        save_adapter_weight(model, config, save_dir)
