# ASPEN: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2023 All Rights Reserved.
#
# Github:  https://github.com/TUDB-Labs/multi-lora-fine-tune

import datetime
import argparse
import torch
import aspen
import os

parser = argparse.ArgumentParser(description='ASPEN main program')
parser.add_argument('--model_name_or_path', type=str, help='Path to or name of base model')
parser.add_argument('--device', type=str, default='cuda:0', help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--log', type=bool, default=True, help='Turn on or off log, default is true')

args = parser.parse_args()


def log(msg: str):
    if args.log:
        print('[%s] ASPEN: %s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


if torch.cuda.is_available():
    log('NVIDIA CUDA initialized successfully.')
    log('Total %i GPU(s) detected.' % torch.cuda.device_count())
else:
    print('ASPEN requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
    exit(-1)


def prep_llm():
    args = aspen.LlamaModelArgs()
    tokenizer = aspen.Tokenizer(args.model_name_or_path + os.sep + 'tokenizer.model')
    tokenizer.pad_id_ = 0
    args.max_seq_len_ = 4096
    args.device = args.device
    args.vocab_size_ = tokenizer.n_words_
    args.pad_id_ = tokenizer.pad_id_
    args.n_heads_ = 32
    model = aspen.LlamaModel(args)
    aspen.load_llama_tf_weight(model, args.model_name_or_path, args.device)
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = prep_llm()