import os
# import args
import openai
import pandas as pd
import json
# openai.api_key = "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" # gpt-4


openai.api_base = "https://yuelyu.openai.azure.com//"
openai.api_key = "33590bee8e1443f39ea87f2041a771af"
openai.api_type = "azure"
openai.api_version = "2023-05-15"

deployment_name = "Yuelyu"

# read i2b2 use the first 20 as the example 
def annotation():
    sleep_temp = "/home/yuj49/ConText/ConText_LLM/sleep_annotation/results.csv"
    data = pd.read_csv(sleep_temp)
    non_na_data = data[data['labels_x'].notna()]
    filtered_data = non_na_data[
        ~non_na_data['labels_x'].str.contains(' ') &
        ~non_na_data['labels_x'].str.contains('fam') &
        ~non_na_data['labels_x'].str.contains('his')
    ]
    # filtered_data = data[data['labels_x'].astype(str).notna() & (~data['labels_x'].astype(str).str.contains(' '))]

    result_21 = filtered_data[['file', 'concept', 'labels_x', 'text']][:21]
    label_mapping = {
        'hyp': 'hypothetical',
        'his': 'historical',
        'neg': 'negated',
        'pos': 'possible',
        'fam': 'family'
        }

    result_21['labels_x'] = result_21['labels_x'].map(label_mapping)
    data_list = result_21.to_dict('records')
    json_list = []

    for item in data_list:
        json_item = {
            "entity": item['concept'],
            "label": item['labels_x'],
            "text": item['text'],
            "file_name": item['file']
        }
        res = chatgpt_tookit(item['concept'],item['text']) # 287 time call api
        print(res)
        json_item["res"] = res
        json_list.append(json_item)

    with open('chatgpt_sleep_cot.json', 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

        
def chatgpt_tookit(condition, content):
    # condition = "burst of atrial fibrillation"
    try:
        # response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        # messages=[
        #         {"role": "system", "content": f"You are an annotation tool tasked with annotating contextual features of snoring from medical records.\
        #         There are five labels that must be appended to each text: Family, Historical, Hypothetical, Negated, Possible.\
        #         Indicate 'True' or 'False' for all five labels.\
        #         Label Family as True when the person who {condition} is a family member of the patient.\
        #         Label Historical as True if the patient's {condition} is a historical condition.\
        #         Label Hypothetical as True if the {condition} is mentioned in a hypothetical statement e.g. 'told patient to return if he continues {condition}'.\
        #         Label Negated as True when the condition is modified by a negative word e.g. 'denies {condition}'.\
        #         Label Possible as true if the condition is not confirmed but is being considered."}, # burst of atrial fibrillation
        #         {"role": "user", "content": content},
        #     ]
        # )
        new_template = f"You are an annotation tool tasked with annotating contextual features of snoring from medical records. \
  Five labels must be appended to each text: Family, Historical, Hypothetical, Negated, and Possible. \
  Indicate 'True' or 'False' for all five labels. \
  1. Label Family as True when the person who {condition} is a family member of the patient. \
  2. Label Historical as True if the patient's {condition} is a historical condition. \
    For exmaple:'history:     past medical history:   diagnosis date     cataract      disorder of skin or subcutaneous tissue      hypothyroidism      nervous system complication      sleep apnea      family history   problem relation age of onset     dizziness biological mother      hearing loss biological mother      migraines biological mother      dizziness biological father      hearing loss biological father      stroke biological father      heart disease biological father      hearing loss brother      sleep apnea brother      sleep apnea brother      sleep apnea brother      other daughter         skin cancer, unknown type     history reviewed.' In this sentence, this have the history and family for the term in the term of 'sleep apnea' So the label 'history: True' and 'family: True' \
  3. Label Hypothetical as True if the {condition} is mentioned in a hypothetical statement e.g. 'told patient to return if he continues {condition}'. \
    For example: '1. Docusate Sodium 100 mg Capsule Sig : One ( 1 ) Capsule PO TID ( 3 times a day ) as needed for Constipation .' For the entity 'constipation' is labeled “Hypothetical: True” because the context”needed for Constipation” that means this may needed.\
  4. Label Negated as True when the condition is modified by a negative word e.g. 'denies {condition}'. \
    For example:'He is not ambulatory and is very deconditioned secondary to several complications within the past two months related to an AICD pocket infection.' in this sentence the entity 'several complications' is labeled: 'Negated: False' because this example shows the patient have several complications because he get  AICD pocket infection.\
    And for the sentence'Patient recorded as having No Known Allergies to Drugs' for the enetity:'known allergies to drugs' labeled 'Negative: True'\
  5. Label Possible as true if the condition is not confirmed but is being considered. \
    For example: 'Question of Tyrone eophagus .' The entity 'Tyrone esophagus' labeled 'possible: True' here because it seems to question the Tyrone esophagus\
  Your output format should follow the structure like 'Family: False Historical: False Hypothetical: False Negated: False Possible: False'. \
    Please don't generate extra things other than the format I give you.\
  Here is the input sentence: {content}.And here is the entity you need to label: {condition}"   

        # start_phrase = 'Write a tagline for an ice cream shop. '
        response = openai.Completion.create(engine=deployment_name, prompt=new_template, max_tokens=400)
        text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    except openai.error.APIError:
        print("server error encountered, waiting 5 seconds...")
        time.sleep(5)
    except openai.error.RateLimitError:
        print("requests are being sent too fast, waiting 30 seconds...")
        time.sleep(30)
    except openai.error.InvalidRequestError:
        print("got InvalidRequestError, skipping this prompt")
        # break
    res = text
    # print(res)
    return res



if __name__ == "__main__":
    annotation()
    # chatgpt_tookit()