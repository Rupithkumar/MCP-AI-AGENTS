# Retrieval-augmented generation basics

## What RAG is
Retrieval-augmented generation (RAG) answers a question in two stages. First, the system retrieves passages from a document collection that are relevant to the question. Second, it gives those passages to a language model, which writes an answer grounded in them. RAG lets a model use private or recent information without retraining, and it reduces hallucination because the model can quote evidence instead of relying on memory.

## The pipeline
A typical RAG pipeline has five stages: load the documents, chunk them into passages, index the chunks, retrieve the best chunks for each question, and generate the answer. The first three stages run ahead of time, while retrieval and generation run for every question. Indexing should be incremental so that only changed files are processed again.

## Citations
A good RAG assistant cites the passage behind each claim, for example with a file name and chunk number. Citations let the reader verify the answer, and they make failures easy to spot, because a claim with no supporting passage is suspect. An answer should only cite passages that were actually retrieved.

## Typical failure modes
Retrieval can return the wrong chunk, or miss the right one entirely. Chunks that are too small lose the surrounding context, while chunks that are too large dilute the relevant sentence among unrelated text. The model can also ignore the retrieved context and answer from memory, or invent a citation that does not exist.

## Knowing when not to answer
When the collection does not contain the answer, the assistant should say that it could not find it. This behaviour is called abstention. A system that always produces a confident answer is dangerous, so abstention must be tested explicitly with questions that the documents cannot answer.
