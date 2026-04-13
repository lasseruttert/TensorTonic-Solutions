import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        
        self.word_to_id[self.pad_token] = self.vocab_size
        self.id_to_word[self.vocab_size] = self.pad_token
        self.vocab_size += 1
        
        self.word_to_id[self.unk_token] = self.vocab_size
        self.id_to_word[self.vocab_size] = self.unk_token
        self.vocab_size += 1

        self.word_to_id[self.bos_token] = self.vocab_size
        self.id_to_word[self.vocab_size] = self.bos_token
        self.vocab_size += 1
        
        self.word_to_id[self.eos_token] = self.vocab_size
        self.id_to_word[self.vocab_size] = self.eos_token
        self.vocab_size += 1

        
        
        all_words = []
        for text in texts:
            for word in text.split():
                all_words.append(word.lower())
        
        all_words = list(set(all_words))
        
        for w in sorted(all_words):
            self.word_to_id[w] = self.vocab_size
            self.id_to_word[self.vocab_size] = w
            self.vocab_size +=1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = []
        
        t = text.lower()
        for word in t.split():

            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])

        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words_as_list = []
        for id in ids:
            words_as_list.append(self.id_to_word.get(id, self.unk_token))
        words_as_str = " ".join(words_as_list)
        return words_as_str

        
        
