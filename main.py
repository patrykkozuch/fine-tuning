from nlp.transformer.transformer import Transformer

a = Transformer(
    vocab_size=30522,
    n_blocks=6,
    num_heads=8,
    seq_len=512,
    d_model=512,
    d_ff=2048
)