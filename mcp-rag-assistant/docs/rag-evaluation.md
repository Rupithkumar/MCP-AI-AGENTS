# Evaluating RAG systems

## Retrieval metrics
Hit rate at k asks whether at least one relevant document appears in the top k results. Mean reciprocal rank (MRR) is the average, over all questions, of one divided by the rank of the first relevant result, so finding the answer at rank 1 scores 1.0 and at rank 4 scores 0.25. Recall at k measures the share of all relevant passages that were retrieved, and nDCG rewards putting the most relevant passages first.

## Generation metrics
Faithfulness, also called groundedness, checks whether every claim in the answer is supported by the retrieved passages. Answer relevance checks that the answer addresses the question. Citation accuracy checks that the cited sources exist and really contain the claim.

## Include unanswerable questions
A useful test set contains questions that the documents cannot answer. For these, the correct behaviour is abstention: the assistant should say it could not find the answer. Without such questions, a system that always invents an answer would look perfect.

## Judging answers
Keyword checks, which look for required facts in the answer, are cheap and repeatable but brittle, because a correct answer can use different words. Using a language model as a judge scales better but can be biased and noisy, so its verdicts should be spot-checked by a person.

## Practical advice
A golden dataset of 30 to 100 questions is a good start, and each question should record the expected source documents. Track cost and latency next to quality, for example tokens and seconds per answer, because a small accuracy gain can come with a large cost increase. Change one thing at a time, such as the chunk size or the retrieval method, and re-run the same evaluation so that differences can be attributed to the change.
