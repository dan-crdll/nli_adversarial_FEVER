# NLI Task with Adversarial Data Augmentation
This is part of the Multi Lingual Natural Language Processing exam of year 2023/2024 in M.Sc. Artificial Intelligence and Robotics.

## Given Task
Design and implement a transformer-based model to perform Natural Language Inference on a subset of FEVER Dataset and in Adversarial Test set.

## Report
To have a more comprehensive insight on the proposed solution and data augmentation pipeline please refer to [MLNLP Adversarial Task Report](report.pdf)

## Approach
Model based on a finetuned distilBERT model (encoding head) along with a MLP classifier. It is also required to augment the data in order to perform better on the adversarial test.
 
## Data Augmentation
The data augmentation pipeline consists of two steps:

1. Premises and Hypotheses editing with synonyms substitution of adjectives, nouns, verbs, and adverbs;
2. Neutral hypotheses generation with GPT-2 pretrained model
