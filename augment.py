import torch
from nltk.corpus import wordnet as wn
import nltk 
from datasets import load_dataset
from transformers import pipeline
from tqdm.auto import tqdm


def reconstruct(entry, part):
    """
    Function to perform the synonyms substitution using the listed nltk synsets on wordnet
    
    params:
        - entry: the entry in the dataset
        - part: ['premise' | 'hypothesis']
    """
    sentence = []
    for w in entry['wsd'][part]:
        word = w['text']
        if w['pos'] == 'ADV' or w['pos'] == 'VERB' or w['pos'] == 'NOUN':   # substitute adverbs, verbs and nouns with randomly selected synonyms
            lemmas = wn.synset(w['nltkSynset']).lemmas()
            word = lemmas[torch.randint(0, len(lemmas), (1,))].name()
        sentence.append(f"{word.replace('_', ' ')} ")
    sentence = "".join(sentence)
    return sentence


def augment():
    nltk.download('wordnet')
    ds = load_dataset('tommasobonomo/sem_augmented_fever_nli')
    ds = ds['train']
    print('Dataset loaded') 
    
    dataset = []
    
    gpt_model = pipeline("text-generation", model="openai-community/gpt2")

    for entry in tqdm(ds):
        neutralize = False
        try:
            modify = torch.randn((1,)).item()
            if not (modify > 0.7 or (entry['label'] == 'NEUTRAL' and modify > 0.2) or (entry['label'] == 'ENTAILMENT' and modify > 0.9)):
                continue
            elif entry['label'] == 'ENTAILMENT' and torch.randn((1, )) > 0.5:
                neutralize = True
            
            if not neutralize:
                premise = reconstruct(entry, 'premise')
                hypothesis = reconstruct(entry, 'hypothesis')

                label = entry['label']
            else:
                premise = reconstruct(entry, 'premise')
                # generate the neutral hypothesis based on a truncated portion of the premise
                hypothesis = gpt_model(entry['premise'][:100], pad_token_id=50256)[0]['generated_text']
                
                label = 'NEUTRAL'
                
            
            dataset.append({
                'premise': premise, 
                'hypothesis': hypothesis,
                'label': label
            })
        except:
            continue

    torch.save(dataset, './augmented_train_set.pth')
    print('Dataset augmented and saves... here an example: ')
    print(dataset[torch.randint(0, len(dataset), (1,))])
    print(f"{len(dataset)} sentences have been augmented.")
    
    
if __name__ == "__main__":
    augment()