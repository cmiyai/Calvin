from typing import Iterable, Iterator
import pickle
import regex as re

class Tokenizer():
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens or [] # easier later for check if specialtoken inside
        self.token_to_id = {tok: i for i, tok in self.vocab.items()} # gives us reverse hashing for token
    
    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str, special_tokens=None):
        """
        Constructs and returns a Tokenizer instance from serialized vocabulary and merge files.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        # merges
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab,merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        '''
        Given input text, merge until we get something that is tokenizable
        '''

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if self.special_tokens:
            delim = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            chunks = re.split(f"({delim})", text)
        else:
            chunks = [text]


        total_tok = []
        for chunk in chunks:
            if chunk == "":
                continue
            if chunk in self.special_tokens:
                chunk = chunk.encode("utf-8")
                total_tok.append(chunk)
            else:
                pretokens = re.findall(PAT, chunk) # chunked
                # for each pretoken: bytes → merge → IDs
                for pretok in pretokens:
                    pretok = [bytes([b]) for b in pretok.encode("utf-8")]
                    mergable = ["bro"]
                    while mergable:
                        mergable = []

                        for i in range(len(pretok) - 1):
                            pair = (pretok[i], pretok[i+1])
                            if pair in self.merge_rank:
                                mergable.append(pair)
                        if not mergable:
                            break
                        merge_byte = min(mergable, key=self.merge_rank.get)

                        newtok = []
                        i = 0 # iterate over one by one to replace adjacent
                        while i < len(pretok):
                            if i < len(pretok) - 1 and (pretok[i], pretok[i + 1]) == merge_byte:
                                newtok.append(pretok[i] + pretok[i + 1])
                                i += 2
                            else:
                                newtok.append(pretok[i])
                                i += 1
                        pretok = newtok # updated with our merged pair
                    
                    total_tok.extend(pretok)
        # encode all at last
        encoded = [self.token_to_id[p] for p in total_tok]
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Return a generator that yields token IDS for memory efficient tokenization of large files
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        decode from encoded vocab
        """
        total_bytes = []
        for id in ids:
            total_bytes.append(self.vocab[id])
        decoded = b"".join(total_bytes).decode("utf-8", errors="replace")
        return decoded

if __name__ == "__main__":
    text = "the cat ate"
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized = re.findall(PAT, text)
    print(pretokenized)