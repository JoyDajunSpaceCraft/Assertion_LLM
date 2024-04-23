# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from llama import Llama, Dialog
# Zero shot
# Ignore warnings

from typing import List, Optional
import fire

from transformers import logging, pipeline, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")


tokenizer.pad_token = tokenizer.eos_token
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model

import json
from datasets import load_metric

# Load the Rouge metric
rouge = load_metric("rouge")
val_file = "/home/yuj49/ConText/ConText_LLM/chatgpt_i2b2_plain.json"
# val_file = "/home/yuj49/ConText/ConText_LLM/chatgpt_sleep_21.json"
with open(val_file) as f:
    data = json.load(f)

prompts_data = []
golden_stands = []

for item in data:
  prompt_string = f"You are an annotation tool tasked with annotating contextual features of snoring from medical records.\
  There are five labels that must be appended to each text: Family, Historical, Hypothetical, Negated, Possible.\
  Indicate 'True' or 'False' for all five labels.\
  Label Family as True when the person who {item['entity']} is a family member of the patient.\
  Label Historical as True if the patient's {item['entity']} is a historical condition.\
  Label Hypothetical as True if the {item['entity']} is mentioned in a hypothetical statement e.g. 'told patient to return if he continues {item['entity']}'.\
  Label Negated as True when the condition is modified by a negative word e.g. 'denies {item['entity']}'.\
  Label Possible as true if the condition is not confirmed but is being considered. \
  Your output format should follow the structure like 'Family: False\nHistorical: False\nHypothetical: False\nNegated: False\nPossible: False'. Please don't generate extra things other than the format I give you. \n\
  Here is the input sentence: {item['text']}. "
  prompts_data.append(prompt_string)
  golden_stands.append(item['label'])
# from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments

generated_texts = []
# val_dataset = Dataset.from_dict(val_data)
for i in prompts_data:
   
   # pipe = pipeline(task="text-generation", model="/home/yuj49/llama2_7B", tokenizer=tokenizer, max_length=400)
   pipe = pipeline(task="text-generation", model="/home/yuj49/llama2_13B", tokenizer=tokenizer, max_length=1000)
  
   # pipe = pipeline(task="text-generation", model="NousResearch/Llama-2-7b-chat-hf", tokenizer=tokenizer, max_length=400)
   result = pipe(i)
   generated_text = result[0]['generated_text']
   print(generated_text)
   generated_text = generated_text[len(i):]
   print(generated_text)
   generated_texts.append(generated_text)

res_list = []
for item, gen in zip(data, generated_texts):
   item["llama_res"] = gen
   res_list.append(item)
with open("llama2_i2b2_13B.json", "w") as f:
   json.dump(res_list, f)

# scores = rouge.compute(predictions=generated_texts, references=golden_stands)
# print(scores)



# import os
# import pandas as pd
# from sklearn.metrics import precision_score, recall_score, f1_score


# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     local_rank: int = 0,
#     temperature: float = 0.6,
#     top_p: float = 0.9,
#     max_seq_len: int = 512,
#     max_batch_size: int = 8,
#     max_gen_len: Optional[int] = None,
# ):
#     """
#     Entry point of the program for generating text using a pretrained model.

#     Args:
#         ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
#         tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
#         temperature (float, optional): The temperature value for controlling randomness in generation.
#             Defaults to 0.6.
#         top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
#             Defaults to 0.9.
#         max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
#         max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
#         max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
#             set to the model's max sequence length. Defaults to None.
#     """
#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#     )
    
#     samples = [("""Patient snores due to osa.""",
#     """Family: False
# Historical: False
# Hypothetical: False
# Negated: False
# Possible: False""")]
#     run_annotation_task(generator, max_gen_len, temperature, top_p, snoring=False, samples=samples)

#     '''dialogs: List[Dialog] = [
#         [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ]
#     ]
#     results = generator.chat_completion(
#         dialogs,  # type: ignore
#         max_gen_len=max_gen_len,
#         temperature=temperature,
#         top_p=top_p,
#     )

#     for dialog, result in zip(dialogs, results):
#         for msg in dialog:
#             print(f"{msg['role'].capitalize()}: {msg['content']}\n")
#         print(
#             f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
#         )
#         print("\n==================================\n")'''

# snoring_system_message = """You are an annotation tool tasked with annotating contextual features of snoring from medical records.
# There are five labels that must be appended to each text: Family, Historical, Hypothetical, Negated, Possible.
# Indicate "True" or "False" for all five labels.
# Label Family as True when the person who snores is a family member of the patient.
# Label Historical as True if the patient's snoring is a historical condition.
# Label Hypothetical as True if the snoring is mentioned in a hypothetical statement e.g. "told patient to return if he continues snoring".
# Label Negated as True when the condition is modified by a negative word e.g. "denies snoring".
# Label Possible as true if the condition is not confirmed but is being considered.
# """

# sleep_apnea_system_message = """You are an annotation tool tasked with annotating contextual features of sleep apnea from medical records.
# There are five labels that must be appended to each text: Family, Historical, Hypothetical, Negated, Possible.
# Indicate "True" or "False" for all five labels.
# Label Family as True when the person with sleep apnea is a family member of the patient.
# Label Historical as True if the patient's sleep apnea is a historical condition.
# Label Hypothetical as True if the snoring is mentioned in a hypothetical statement e.g. "told patient to consider cpap if she develops osa".
# Label Negated as True when the condition is modified by a negative word e.g. "denies sleep apnea".
# Label Possible as true if the condition is not confirmed but is being considered.
# """


# def check_filename(file):
#    return file[-4:] == ".txt"

# def load_records():
#    return pd.read_csv('results.csv')

# def segment_row(row):
#    '''for line in row.split('\n'):
#       if 'snor' in line:
#          yield line'''
#    return [row[1][7]]

# # if samples are given, will make a fewshot prompt
# def make_annotation_prompt(text, snoring=True, samples=None):
#    if snoring:
#       system_message = snoring_system_message
#    else:
#       system_message = sleep_apnea_system_message
#    messages = [{"role": "system", "content": system_message}]
#    if samples:
#       for prompt, response in samples:
#          messages.append({"role": "user", "content": prompt})
#          messages.append({"role": "assistant", "content": response})
#    messages.append({"role": "user", "content": text})
#    return messages

# def process_output(text):
#    labels = []
#    try:
#       if text.split('Family: ')[1].split()[0] == 'True':
#          labels.append('fam')
#       if text.split('Historical: ')[1].split()[0] == 'True':
#          labels.append('his')
#       if text.split('Hypothetical: ')[1].split()[0] == 'True':
#          labels.append('hyp')
#       if text.split('Negated: ')[1].split()[0] == 'True':
#          labels.append('neg')
#       if text.split('Possible: ')[1].split()[0] == 'True':
#          labels.append('pos')
#       return ' '.join(labels)
#    except IndexError:
#       return None

# def compare_labels(true, pred):
#    label_list = []
#    for label in ['fam', 'his', 'hyp', 'neg', 'pos']:
#       true_labels = []
#       pred_labels = []
#       for t, p in zip(true, pred):
#          true_labels.append(1 if not isinstance(t, float) and label in t else 0)
#          pred_labels.append(1 if p != None and bool(len(p.strip())) and label in p else 0)
#       label_list.append((true_labels, pred_labels))
#    return label_list

# def print_stats(true_labels, pred_labels):
#    with open('stats.txt', 'a') as stats_file:
#       stats_file.write(str(true_labels) + '\n')
#       stats_file.write(str(pred_labels) + '\n')
#       stats_file.write(str(precision_score(true_labels, pred_labels)) + '\n')
#       stats_file.write(str(recall_score(true_labels, pred_labels)) + '\n')
#       stats_file.write(str(f1_score(true_labels, pred_labels)) + '\n')

# def run_annotation_task(generator, max_gen_len, temperature, top_p, snoring=True, samples=None):
#    df = load_records()
#    true_labels = []
#    pred_labels = []
#    for row in df.iterrows():
#       if snoring == (row[1][3] != 'snoring'):
#          continue
#       for segment in segment_row(row):
#          print(segment[:50])
#          prompt_messages = make_annotation_prompt(segment, samples=samples, snoring=snoring)
#          try:
#             result = generator.chat_completion(
#                [prompt_messages],  # type: ignore
#                max_gen_len=max_gen_len,
#                temperature=temperature,
#                top_p=top_p,
#             )[0]
#          except AssertionError:
#             with open('failed.txt', 'w+') as failed:
#                failed.write(segment + '\n')
#             continue
#          print(result)
#          output_text = result['generation']['content']
#          with open('annotation_results_context.txt', 'a', encoding='utf8') as output:
#             output.write('RECORD TEXT: ' + segment + '\n\nRESPONSE: ')
#             output.write(output_text + '\n\n')
#          labels = process_output(output_text)
#          pred_labels.append(labels)
#          true_labels.append(row[1][4])
#          print(labels)
#          print(row[1][4])
#    for pair, name in zip(compare_labels(true_labels, pred_labels), ['fam', 'hist', 'hypo', 'neg', 'poss']):
#       print_stats(pair[0], pair[1])

# if __name__ == "__main__":
#     fire.Fire(main)
