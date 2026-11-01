"""The byte-level BPE tokenizer: it must be lossless."""

from superfluid.bpe import train_bpe


def test_a_tiny_tokenizer_round_trips():
    text = ("Alice was beginning to get very tired of sitting by her sister "
            "on the bank.  'What a fine day!' she said.")
    tok = train_bpe(text, vocab_size=300, verbose=False)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_unseen_text_still_round_trips():
    """The byte fallback means nothing is ever unrepresentable."""
    tok = train_bpe("hello world " * 20, vocab_size=280, verbose=False)
    weird = "naïve café — 日本語 🎉"
    assert tok.decode(tok.encode(weird)) == weird


def test_the_vocabulary_is_the_size_we_asked_for():
    tok = train_bpe("a b c " * 200, vocab_size=300, verbose=False)
    assert tok.vocab_size <= 300
    assert tok.vocab_size >= 256
