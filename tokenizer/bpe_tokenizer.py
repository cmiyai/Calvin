from collections import defaultdict
import regex as re
import pdb

from tqdm import tqdm

class BPETrainer:
    def __init__(self, vocab_size: int = 256, special_tokens: list[str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

    def merge_pair_byte(self, pair: tuple[bytes, bytes]) -> bytes:
        return pair[0] + pair[1]

    def merge_pair_tuple(self, old_tuple: tuple[bytes, ...], most_freq: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        '''
        Helper Function that merges the most freq pair in old tuple
        Returns the new tuple with the pair merged
        '''
        merged_tok = most_freq[0] + most_freq[1]
        new = []
        i = 0
        n = len(old_tuple)
        while i < n:
            pair = (old_tuple[i], old_tuple[i+1]) if i < n - 1 else None
            if pair == most_freq:
                new.append(merged_tok)
                i += 2
            else:
                new.append(old_tuple[i])
                i += 1
        return tuple(new)
    
    def count_pretokens(self, text:str) -> dict[tuple[bytes, ...], int]:
        frequency = defaultdict(int)
        if self.special_tokens:
            delim = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            segments = re.split(delim, text)
        else:
            segments = [ text]
        
        # pretokenize the segment
        for segment in segments:
            for t in self.pretokenize(segment):
                raw_bytes = t.encode("utf-8")
                tok = tuple(bytes([b]) for b in raw_bytes)
                frequency[tok] += 1
        return frequency
    
    def run_merges(
            self,
            frequency: dict[tuple[bytes, ...], int],
            progress: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''
        Loops through w/ bpe iterations
        '''
        pair_freq = defaultdict(int)
        pair_to_tuples = defaultdict(set)
        self.merges = list()
        self.vocab = {i: bytes([i]) for i in range(256)}
        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")

        # optimized pretokenizer searching through the pairs that matter
        for key, count in frequency.items():
            for i in range(1, len(key)):
                pair = (key[i-1], key[i]) # tuple of our token pairs
                pair_to_tuples[pair].add(key)
                pair_freq[pair] += count
        
        n_merges = self.vocab_size - len(self.vocab)
        bar = tqdm(total=n_merges, desc="merge", disable=not progress)
        
        while len(self.vocab) < self.vocab_size:
            if not pair_freq:
                break
            
            # find most frequent byte pairing
            most_freq = max(pair_freq, key=lambda p: (pair_freq[p], p))


            if pair_freq[most_freq] <= 0:
                break
            merged = self.merge_pair_byte(most_freq)
            n = len(self.vocab)
            self.vocab[n] = merged
            self.merges.append(most_freq)
            #print(f"mergelist: {self.merges}") # debug
            affected_tuples = list(pair_to_tuples[most_freq])

            # change affected tuples and shift encoding
            for old_tuple in affected_tuples:
                count = frequency.pop(old_tuple)
                # add edge case?
                new_tuple = self.merge_pair_tuple(old_tuple, most_freq)
                frequency[new_tuple] = count

                # we have to remove the new pairs messed up by the merge
                # NOTE THAT THE OLD PAIRS STILL STAY IN THE VOCABULARY
                for i in range(len(old_tuple) - 1):
                    old_pair = (old_tuple[i], old_tuple[i + 1])
                    pair_freq[old_pair] -= count
                    pair_to_tuples[old_pair].discard(old_tuple)
                
                # add the new tokenization split
                for i in range(len(new_tuple) - 1):
                    new_pair = (new_tuple[i], new_tuple[i + 1])
                    pair_freq[new_pair] += count
                    pair_to_tuples[new_pair].add(new_tuple)
            bar.update(1)
        bar.close()
        return (self.vocab, self.merges)

    def pretokenize(self, text: str) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return re.findall(PAT, text)

    def train(self, input_text: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''
        BPE tokenizer training
        '''
        frequency = self.count_pretokens(input_text)
        
        tokenized = self.run_merges(frequency)
        return tokenized


if __name__ == "__main__":
    test_input = (
        "The brown fox jumped over the lazy dog. "
        "The brown fox jumped over the lazy dog! "
        "The brown fox, the brown fox, the brown fox jumped higher and higher. "
        "Did the lazy dog see the brown fox jump? "
        "Yes, the lazy dog saw the brown fox jump over the lazy dog."
        "The quick brown fox jumps over the lazy dog."
    )
    
    trainer = BPETrainer(vocab_size=400, special_tokens=["dog", " dog"])
    vocab, merges = trainer.train(test_input)
    print(vocab, merges)