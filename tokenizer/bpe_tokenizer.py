from collections import defaultdict
import regex as re
import pdb
import heapq

from tqdm import tqdm

class Node:
        """Node for tokens inside words as doubly linked list"""
        __slots__ = ("token", "prev", "next", "deleted")
        def __init__(self, token: bytes):
            self.token = token
            self.prev: int = -1
            self.next: int = -1
            self.deleted: bool = False

class BPETrainer:
    def __init__(self, vocab_size: int = 256, special_tokens: list[str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

    def merge_pair_byte(self, pair: tuple[bytes, bytes]) -> bytes:
        return pair[0] + pair[1]
    
    
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
        Rewrite on every merge with doubly linked-list
        '''
        pair_freq = defaultdict(int)
        self.merges = list()
        self.vocab = {i: bytes([i]) for i in range(256)}
        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")

        words = []
        word_freqs = []
        pair_locations = defaultdict(list)

        for word_tuple, count in frequency.items():
            if not word_tuple:
                continue
            w_id = len(words)
            word_freqs.append(count)
            nodes = [Node(tok) for tok in word_tuple]
            n_toks = len(nodes)
            for i in range(n_toks):
                if i > 0:
                    nodes[i].prev = i - 1
                if i < n_toks - 1:
                    nodes[i].next = i + 1
                    pair = (nodes[i].token, nodes[i + 1].token)
                    pair_freq[pair] += count
                    pair_locations[pair].append((w_id, i))
            words.append(nodes)
        
        n_merges = self.vocab_size - len(self.vocab)
        bar = tqdm(total=n_merges, desc="merge", disable=not progress)

        heap = [(-count, pair) for pair, count in pair_freq.items()]
        heapq.heapify(heap)
        
        while len(self.vocab) < self.vocab_size:
            if not pair_freq:
                break
            
            # find most frequent byte pairing
            # Use heap for log(k) merges
            most_freq = None
            while heap:
                neg_count, candidate_pair = heapq.heappop(heap)
                if -neg_count == pair_freq.get(candidate_pair, 0) and -neg_count > 0:
                    most_freq = candidate_pair
                    break

            if most_freq is None:
                break


            if pair_freq[most_freq] <= 0:
                break
            merged = self.merge_pair_byte(most_freq)
            n = len(self.vocab)
            self.vocab[n] = merged
            self.merges.append(most_freq)
            #print(f"mergelist: {self.merges}") # debug
            locs = pair_locations.pop(most_freq, [])
            for w_id, i in locs:
                nodes = words[w_id]
                freq = word_freqs[w_id]

                curr = nodes[i]
                if curr.deleted or curr.next == -1:
                    continue

                nxt = nodes[curr.next]
                if nxt.deleted or (curr.token, nxt.token) != most_freq:
                    continue

                p_idx = curr.prev
                n_idx = nxt.next

                # Capture original tokens before modifying curr
                old_curr_tok = curr.token
                old_nxt_tok = nxt.token

                # 1. Decrement counts of broken adjacent pairs using original tokens
                if p_idx != -1:
                    pair_freq[(nodes[p_idx].token, old_curr_tok)] -= freq
                if n_idx != -1:
                    pair_freq[(old_nxt_tok, nodes[n_idx].token)] -= freq

                # 2. Splice out nxt in O(1)
                curr.token = merged
                curr.next = n_idx
                nxt.deleted = True
                if n_idx != -1:
                    nodes[n_idx].prev = i

                # 3. Add newly created adjacent pairs with merged token
                if p_idx != -1:
                    left_pair = (nodes[p_idx].token, merged)
                    pair_freq[left_pair] += freq
                    pair_locations[left_pair].append((w_id, p_idx))
                    heapq.heappush(heap, (-pair_freq[left_pair], left_pair))

                if n_idx != -1:
                    right_pair = (merged, nodes[n_idx].token)
                    pair_freq[right_pair] += freq
                    pair_locations[right_pair].append((w_id, i))
                    heapq.heappush(heap, (-pair_freq[right_pair], right_pair))

            del pair_freq[most_freq]
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