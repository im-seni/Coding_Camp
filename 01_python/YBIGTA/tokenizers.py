import re, collections
from typing import Optional, Union, List, Tuple, Dict

class Preprocessing:
    def preprocess(self, text: str) -> (dict, dict):
        text = re.sub(r"[^0-9a-zA-Z]", " ", text.lower())

        all_voca = text.strip().split()
        spaced_vocab= collections.Counter(' '.join(word) for word in all_voca)

        char_vocab = collections.Counter(char for word in all_voca for char in word)

        total_vocab = spaced_vocab + char_vocab

        return total_vocab

class TokenizerBase:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        if isinstance(corpus, List):
            self.corpus = " ".join(corpus)
        else:
            self.corpus = corpus

    def add_corpus(self, corpus: Union[List[str], str]) -> None:

        if isinstance(corpus, List):
            self.corpus += " ".join(corpus)
        else:
            self.corpus += " " + corpus   

    #About tokenization - padding and truncation
    def apply_truncation(self, tokens: List[int], max_length: Optional[int]) -> List[int]:
            if max_length != None and len(tokens) > max_length:
                return tokens[:max_length]
            return tokens  

    def apply_padding(self, tokens: List[int], max_length: Optional[int],padding_token: int) -> List[int]:
            if len(tokens) < max_length:
                tokens.extend([padding_token] * (max_length - len(tokens)))
            return tokens


class BPETokenizer(TokenizerBase):
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        super().__init__(corpus)
        self.bpe_codes = collections.defaultdict(int)

    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word, freq in vocab.items():
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = freq
        return v_out

    def train(self, n_iter: int) -> None: 
        self.pre = Preprocessing()
        self.vocab = self.pre.preprocess(self.corpus)

        for i in range(n_iter):
            pairs = self.get_stats(self.vocab)
            best = max(pairs, key=pairs.get)
            self.bpe_codes[best] = i
            self.vocab = self.merge_vocab(best, self.vocab)

    def tokenize(
            self, text: Union[List[str], str], 
            padding: bool = False,
            max_length: Optional[int] = None
            ) -> Union[List[List[int]], List[int]]:

        if isinstance(text, list):
            self.text = ' '.join(text)
        else: 
            self.text  = text.strip() 

        self.padding = padding
        self.max_length = max_length 

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}

        self.tokens = []
        for word in self.text.split():
            word = re.sub(r"[^0-9a-zA-Z]", " ", word.lower())
            spaced_word = ' '.join(word)
            self.tokens.extend(spaced_word.split())

        while True:
            new_tokens = []
            whether_merged = False
            skip = False

            for i in range(len(self.tokens)-1):
                if skip: #if current token was merged in previous step
                    skip = False
                    continue
                pair = (self.tokens[i], self.tokens[i+1])

                if pair in self.bpe_codes:
                    new_tokens.append(''.join(pair)) #merge
                    skip = True
                    whether_merged = True
                else:
                    new_tokens.append(self.tokens[i])

            if not skip:
                new_tokens.append(self.tokens[-1])
            self.tokens = new_tokens

            if not whether_merged:
                break


        #token to id 
        self.token_ids = [self.token_to_id.get(token, -1) for token in self.tokens]

        # Truncation 및 Padding 적용
        self.token_ids = self.apply_truncation(self.token_ids, max_length)
        if padding:
            self.token_ids = self.apply_padding(self.token_ids, max_length, padding_token = -2)

        return self.token_ids

    def __call__ (
            self,
            text: Union[List[str], str], 
            padding: bool = False, 
            max_length: Optional[int] = None) -> List[int]:
        return self.tokenize(text, padding, max_length)

class WordTokenizer (TokenizerBase):
    def __init__ (self, corpus: Optional[Union[List[str], str]] = None):
        super().__init__(corpus)

    def train(self, *args, **kwargs) -> None:
        words = re.sub(r"[^0-9a-zA-Z]", " ", self.corpus.lower())
        words = self.corpus.split()
        self.vocab = collections.Counter(word for word in words)

        # Create token to ID mapping
        self.token_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id_to_token = {idx: word for word, idx in self.token_to_id.items()}

    def tokenize(
            self, 
            text: Union[List[str], str], 
            padding: bool = False, 
            max_length: Optional[int] = None) -> List[int]:
        """
        Tokenize a text and return list of token ids.
        """
        if isinstance(text, list):
            self.text = ' '.join(text)
        else:
            self.text = text.strip()

        self.text = self.text.lower()

        self.result = []
        for word in self.text.split():
            word = re.sub(r"[^0-9a-zA-Z]+", '', word)
            word_id = self.token_to_id.get(word, -1)
            self.result.append(word_id)

        # Truncation 및 Padding 적용
        self.result = self.apply_truncation(self.result, max_length)

        # Apply padding
        if padding:
            while len(self.result) < max_length:
                self. result.append(-2)


        return self.result

    def __call__ (
            self,
            text: Union[List[str], str], 
            padding: bool = False, 
            max_length: Optional[int] = None) -> List[int]:
        return self.tokenize(text, padding, max_length)