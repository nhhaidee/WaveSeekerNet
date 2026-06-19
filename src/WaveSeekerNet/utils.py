import logging
import sys
import numpy as np
from Bio import SeqIO

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)


def count_fasta_sequences(fasta_path: str) -> int:
    """
    Fast, zero-dependency helper to count the total number of records
    in a FASTA file by scanning header lines starting with '>'.
    """
    count = 0
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def encode_chunk(
        chunk_seqs: list[str],
        seq_len: int,
        res_l: int,
        convert_ambiguous_to_n: bool
) -> np.ndarray:
    """Helper to vectorized-encode a list of sequences to one-hot format."""
    n_seqs = len(chunk_seqs)

    # 1. Pre-allocate character array filled with 'N' (acts as padding/unknown)
    seq_chars = np.full((n_seqs, seq_len), 'N', dtype='U1')
    for i, seq in enumerate(chunk_seqs):
        curr_len = min(len(seq), seq_len)
        # Ensure uppercase for comparison safety
        seq_chars[i, :curr_len] = list(seq[:curr_len].upper())

    one_hot = np.zeros((n_seqs, res_l, seq_len), dtype=np.float32)

    # 2. Match standard bases
    is_A = (seq_chars == 'A')
    is_C = (seq_chars == 'C')
    is_G = (seq_chars == 'G')
    is_T = (seq_chars == 'T') | (seq_chars == 'U')

    one_hot[:, 0, :] = is_A
    one_hot[:, 1, :] = is_C
    one_hot[:, 2, :] = is_G
    one_hot[:, 3, :] = is_T

    # 3. Identify ambiguous (non-standard) bases
    is_standard = is_A | is_C | is_G | is_T
    is_ambiguous = ~is_standard
    is_N = (seq_chars == 'N')
    is_other_ambiguous = is_ambiguous & ~is_N

    # 4. Map ambiguous bases according to res_l
    if res_l == 5:
        # Index 4: any character that is not A, C, G, T (including N, gaps, etc.)
        one_hot[:, 4, :] = is_ambiguous

    elif res_l == 6:
        # Index 4: N (and other ambiguous bases if convert_ambiguous_to_n is True)
        # Index 5: Other non-standard characters
        if convert_ambiguous_to_n:
            one_hot[:, 4, :] = is_ambiguous
            one_hot[:, 5, :] = 0.0
        else:
            one_hot[:, 4, :] = is_N
            one_hot[:, 5, :] = is_other_ambiguous

    return one_hot


def fasta_to_one_hot(
        fasta_path: str,
        seq_len: int,
        res_l: int = 5,
        convert_ambiguous_to_n: bool = True,
        chunk_size: int = 50000,
        out_filename: str | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Read DNA sequences from a FASTA file and convert to one-hot encoding.

    Uses disk memory-mapping (np.memmap) if out_filename is provided to allow
    processing datasets that exceed available RAM (e.g. 500K+ sequences).

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    seq_len : int
        Target sequence length. Sequences are padded or truncated.
    res_l : int, default 5
        One-hot channel size. If 5: A->0, C->1, G->2, T->3, Ambiguous->4.
    convert_ambiguous_to_n : bool, default True
        Convert all non-A/C/G/T bases to N (index 4 in 5-channel mode).
    chunk_size : int, default 50000
        Number of sequences to process in a single batch to limit RAM spikes.
    out_filename : str or None, default None
        If provided, writes the array directly to a memory-mapped NumPy file.
        Highly recommended for large datasets (e.g. 500k sequences).

    Returns
    -------
    X : np.ndarray
        One-hot encoded array of shape (n_sequences, res_l, seq_len).
        If out_filename is provided, this is a memory-mapped numpy array.
    headers : list of str
        The FASTA headers (IDs) of the encoded sequences in corresponding order.
    """
    # 1. Count sequences in FASTA
    n_seqs = count_fasta_sequences(fasta_path)
    logger.info("Found %d sequences in FASTA file: %s", n_seqs, fasta_path)

    # 2. Setup the output array (disk-backed or in-memory)
    if out_filename:
        logger.info("Initializing disk-backed memory-mapped array at %s...", out_filename)
        fp = np.memmap(out_filename, dtype='float32', mode='w+', shape=(n_seqs, res_l, seq_len))
    else:
        logger.info("Initializing in-memory array (Warning: Make sure you have enough RAM)...")
        fp = np.zeros((n_seqs, res_l, seq_len), dtype=np.float32)

    # 3. Stream and encode
    headers = []
    chunk = []
    chunk_start = 0

    for record in SeqIO.parse(fasta_path, "fasta"):
        headers.append(record.id)
        # Convert record seq to string and upper case immediately for safety
        chunk.append(str(record.seq).upper())

        if len(chunk) == chunk_size:
            chunk_end = chunk_start + len(chunk)
            fp[chunk_start:chunk_end] = encode_chunk(
                chunk, seq_len, res_l, convert_ambiguous_to_n
            )
            logger.info("Encoded records %d to %d...", chunk_start, chunk_end)
            chunk_start = chunk_end
            chunk = []

    # Process remaining records
    if chunk:
        chunk_end = chunk_start + len(chunk)
        fp[chunk_start:chunk_end] = encode_chunk(
            chunk, seq_len, res_l, convert_ambiguous_to_n
        )
        logger.info("Encoded final records %d to %d.", chunk_start, chunk_end)

    if out_filename:
        fp.flush()
        logger.info("Encoding complete. Dataset saved to %s", out_filename)

    return fp, headers