# Retrieval methods

## BM25
BM25 is a keyword-based ranking function. It scores a passage by how often the query terms appear in it, how rare those terms are across the collection, and how long the passage is. Two parameters control its behaviour. The parameter k1 sets term frequency saturation, so that repeating a word many times gives diminishing returns, and it is commonly set between 1.2 and 2.0. The parameter b controls length normalisation, which stops long passages from winning just because they contain more words, and it is commonly 0.75. BM25 is excellent for exact terms and rare identifiers, but it cannot match synonyms.

## Dense retrieval
Dense retrieval converts the query and each passage into embedding vectors and ranks passages by cosine similarity. Because embeddings capture meaning, dense retrieval handles paraphrases and synonyms that share no words with the question. Its weakness is exact identifiers, product codes and unusual names, which embeddings can blur.

## Hybrid retrieval
Hybrid retrieval runs both methods and merges the rankings, so that exact matches and meaning-based matches both contribute. A popular merging method is Reciprocal Rank Fusion. Each passage receives a score of 1 divided by (k plus its rank) from every ranking it appears in, and the scores are added. The constant k is commonly 60. Because only ranks are used, the method needs no normalisation of the different score scales.

## Reranking
A reranker is a second, slower model that re-scores the top 20 to 50 candidates from the first stage. Cross-encoder rerankers read the query and passage together, which makes them more accurate than the first-stage ranking, at the cost of extra latency.
