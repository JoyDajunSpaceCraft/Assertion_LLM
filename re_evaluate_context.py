import json
import argparse
# chatgpt_21 = "/home/yuj49/ConText/ConText_LLM/chatgpt_i2b2_21.json"
# chatgpt_21 = "/home/yuj49/ConText/ConText_LLM/chatgpt_sleep_21.json"
llama2_22 = "/home/yuj49/ConText/ConText_LLM/llama2_i2b2_21.json"
# llama2_22 = "/home/yuj49/ConText/ConText_LLM/llama2_sleep_21.json"

def count_acc(file, llama=False):
    with open(file, "r") as f:
        data = json.load(f)
    count = 0
    for idx, item in enumerate(data):
        # print(idx)
        if llama:
            res = item["llama_res"]
        else:
            res = item["res"]

        label = item["label"]
        res = str.lower(res)
        if "family" not in res and "historical" not in res and "hypothetical" not in res and "negated" not in res and "possible" not in res:
            continue
        if "negated" not in res:
            print("*"*10  + "\n" + "wrong prediction is " + res + "\n" + " label: " + label + "\n text: " + item["text"] + "\n entity: " + item["entity"])
        else:
            if label == "absent" and res.split("negated: ")[1].startswith("true"):
                count+=1
            elif label== "present":
                if "negated: " in res and res.split("negated: ")[1].startswith("false"):
                    count+=1
                elif "negated:" in res and res.split("negated:")[1].startswith("false"):
                    count+=1
            elif label in res and label+": " in res and res.split(label+": ")[1].startswith("true"):
                count+=1
            else:
                print("*"*10  + "\n" + "wrong prediction is " + res + "\n" + " label: " + label + "\n text: " + item["text"] + "\n entity: " + item["entity"])
    print(count)

def count_prf(file, llama=False):
    llama2_22 = "/home/yuj49/ConText/ConText_LLM/llama2_i2b2_21.json"

    with open(file, "r") as f:
        data =  json.load(f)

    # Define the categories
    categories = ["Family", "Historical", "Hypothetical", "Negated", "Possible","Postive"]

    # Initialize counters for each category
    stats = {}
    for cat in categories:
        stats[cat] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    count = 0
    for item in data:
        if llama:
            res = item["llama_res"]
        else:
            res = item["res"]

        label = item["label"]
        if label == "absent":
            label = "negated"
        # if label ==
        res = str.lower(res)

        for cat in categories:
            if cat.lower() in res:
                prediction = res.split(cat.lower() + ": ")[1].startswith("true")
                truth = label == cat.lower()
                if prediction and truth:
                    stats[cat]["TP"] += 1
                elif prediction and not truth:
                    stats[cat]["FP"] += 1
                elif not prediction and truth:
                    stats[cat]["FN"] += 1
                else:
                    stats[cat]["TN"] += 1

    # Calculate Macro-average Recall, Precision, F1-score
    total_recall, total_precision, total_f1 = 0, 0, 0

    for cat in categories:
        TP = stats[cat]["TP"]
        FP = stats[cat]["FP"]
        FN = stats[cat]["FN"]

        recall = TP / (TP + FN) if TP + FN > 0 else 0
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        total_recall += recall
        total_precision += precision
        total_f1 += f1

        print(f"{cat} - Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

    macro_recall = total_recall / len(categories)
    macro_precision = total_precision / len(categories)
    macro_f1 = total_f1 / len(categories)

    print(f"Macro-average Recall: {macro_recall:.4f}")
    print(f"Macro-average Precision: {macro_precision:.4f}")
    print(f"Macro-average F1-score: {macro_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, default="/home/yuj49/ConText/ConText_LLM/llama2_sleep_13B_cot.json"
    )
    parser.add_argument("--llama", type=bool, default=False)
    parser.add_argument("--acc", type=bool, default=False)
    args = parser.parse_args()
    if args.acc:
        count_acc(args.file_path, args.llama)
    else:
        count_prf(args.file_path, args.llama)

if __name__ == "__main__":
    main()