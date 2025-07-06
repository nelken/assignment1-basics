import regex as re
from collections import Counter, defaultdict
import logging
from typing import BinaryIO
import os
import concurrent.futures

logging.basicConfig(level=logging.ERROR)

num_processes = 2


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
   
    return sorted(set(chunk_boundaries))
     
     
def replace_pair(top_item, pretoken):
    c1, c2 = top_item[0]
    result = []
    if c1 not in pretoken: 
        return pretoken
    if c2 not in pretoken: 
        return pretoken
    
    i = 0 

    while i < len(pretoken):
        if pretoken[i] == c1 and i < len(pretoken) - 1  and pretoken[i + 1] == c2:
            result.append(c1 + c2)
            i += 2  # skip both merged tokens
        else:
            result.append(pretoken[i])
            i += 1
    return tuple(result)

def replace(top_item, pretoken_counter):
    return Counter({
        replace_pair(top_item, pretoken): count
        for pretoken, count in pretoken_counter.items()
    })
    

def init_vocab(special_tokens: list[str])->dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")        
    return vocab
    
def split_on_tokens(text, tokens):
    # Escape special regex characters in tokens
    escaped = [re.escape(tok) for tok in tokens]
    pattern = '|'.join(escaped)
    return re.split(pattern, text)


def to_initial_tokens(s: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in s.encode('utf-8'))

def pretokenize(full_text):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretoken_counter = Counter()

    texts = split_on_tokens(full_text, ["<|endoftext|>"])
    for text in texts:
        for match in re.findall(PAT, text):
            byte_tuple = to_initial_tokens(match)
            pretoken_counter[byte_tuple] += 1
    return pretoken_counter        


def train_bpe( 
    input_path: str,
    vocab_size: int = 500, 
    special_tokens: list[str]=["<|endoftext|>"]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab = init_vocab(special_tokens)
    
    # pretokenization
    pretoken_counter = Counter()
    with open(input_path, "r") as f:
        pretoken_counter = pretokenize(f.read())
    logging.debug(pretoken_counter)
    merges = []

    while len(vocab) < vocab_size:
        pairs_counter = Counter()
        for pretoken, count in pretoken_counter.items():                
            pairs = zip(pretoken, pretoken[1:])
            for pair in pairs: 
                pairs_counter[pair] += count

        if not pairs_counter:
            break
        top_item = max(pairs_counter.items(), key=lambda x: (x[1], x[0]))
        new_token = b''.join(top_item[0])
        if not new_token in vocab.values():
            vocab[len(vocab)] = new_token

        merges.append((top_item[0][0], top_item[0][1]))
        pretoken_counter = replace(top_item, pretoken_counter)                        
           
        logging.debug(pretoken_counter)
    logging.debug(vocab, merges)
    for k, v in vocab.items():
        if v not in special_tokens:
            vocab[k] = v                

    return vocab, merges 

