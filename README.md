# Assertion Detection in English EHR Using LLMs

## Introduction

This repository is dedicated to our research on enhancing the extraction of clinical information from English Electronic Health Records (EHR). We focus on assertion detection, a critical step in generating actionable clinical information, which involves categorizing medical entities based on their state such as 'Positive', 'Negative', 'Possible', 'Historical', 'Hypothetical', and 'Family' related conditions.

## Challenge

The challenge of assertion detection in EHRs lies in the unstructured format of clinical notes. Traditional methods have limitations in identifying less frequent assertions and generally perform suboptimally on categories beyond 'Positive' or 'Negative'.

## Our Approach

We introduce advanced reasoning methodologies applied to Large Language Models (LLMs) pre-trained on medical data to improve the detection and categorization of assertions. Our methods include Tree of Thought (ToT), Chain of Thought (CoT), and Self-Consistency (SC), enhanced by Low-Rank Adaptation (LoRA) fine-tuning techniques on the LLama 2 model.

## Contributions

- **Advanced LLMs**: We developed LLMs enhanced with ToT, CoT, and SC to improve assertion detection in medical narratives.
- **LLama 2 Fine-tuning**: Utilizing LoRA, we optimized LLama 2 for greater precision in clinical assertion detection.
- **Specialized Annotations**: We focused on sleep-related disorders, such as sleep apnea and snoring, to demonstrate our fine-tuned LLMs' effectiveness and adaptability.

## Repository Contents

- Annotation Guidelines
- Pre-trained Models
- Fine-tuning Scripts
- Evaluation Metrics
- Example Notebooks


## Usage

Instructions on how to use the models and scripts for assertion detection are provided in the respective directories.

## Citation

If you use the resources provided in this repository, please cite:

```
@article{ji2024assertion,
  title={Assertion Detection Large Language Model In-context Learning LoRA Fine-tuning},
  author={Ji, Yuelyu and Yu, Zeshui and Wang, Yanshan},
  journal={arXiv preprint arXiv:2401.17602},
  year={2024}
}
```

## Acknowledgements

Our work draws upon established frameworks by ConText and i2b2, as well as data from sleep studies. We thank these contributions which were invaluable to our research.

## Contact

For any inquiries, please open an issue on this repository or contact us directly at [yuj49@pitt.edu].
