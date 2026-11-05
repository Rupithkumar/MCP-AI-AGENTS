# Chunking strategies

## Fixed-size chunks
The simplest strategy cuts the text into pieces of a fixed size, commonly 500 to 1,000 characters or 200 to 500 tokens. It is fast and predictable, but it can cut a sentence in half and separate an idea from its explanation.

## Overlap
To reduce the damage from hard boundaries, consecutive chunks usually overlap by roughly 10 to 20 percent. Overlap repeats the end of one chunk at the start of the next, so an idea that straddles a boundary still appears whole in at least one chunk. The cost is a larger index and some duplicated text in the results.

## Sentence-aware chunking
Instead of cutting at an exact character count, sentence-aware chunking packs whole sentences until the size limit is reached. This keeps every chunk readable and makes overlap natural, because the last sentence can simply be repeated.

## Structure-aware chunking
Documents with headings should be split along their structure. Each chunk keeps its section path, such as "Guide > Setup", as metadata. The section title helps retrieval because it names the topic, and it gives readers context when the chunk is shown in a citation.

## Semantic chunking
Semantic chunking places boundaries where the meaning changes, for example where the similarity between neighbouring sentence embeddings drops. It can produce more coherent chunks but needs an embedding model and is slower to build.

## Parent-child retrieval
In parent-child retrieval the system searches over small child chunks, which match precisely, but returns the larger parent section that contains the match. This gives the model more context than the small chunk alone while keeping the search accurate.
