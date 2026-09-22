inclusion of redundant information. To estimate context relevance, given a question q and its context c(q), the LLM extracts a subset of sentences, $S _ { e x t }$ , from c(q) that are crucial to answer q, using the following prompt:

Please extract relevant sentences from the provided context that can potentially help answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information". While extracting candidate sentences you’re not allowed to make any changes to sentences from given context.

The context relevance score is then computed as:

$$
\mathbf { C R } = { \frac { \mathrm { n u m b e r ~ o f ~ e x t r a c t e d ~ s e n t e n c e s } } { \mathrm { t o t a l ~ n u m b e r ~ o f ~ s e n t e n c e s ~ i n ~ } c ( q ) } }\tag{2}
$$

## 4 The WikiEval Dataset

To evaluate the proposed framework, we ideally need examples of question-context-answer triples which are annotated with human judgments. We can then verify to what extent our metrics agree with human assessments of faithfulness, answer relevance and context relevance. Since we are not aware of any publicly available datasets that could be used for this purpose, we created a new dataset, which we refer to as WikiEval4. To construct the dataset, we first selected 50 Wikipedia pages covering events that have happened since the start of 20225. In selecting these pages, we prioritised those with recent edits. For each of the 50 pages, we then asked ChatGPT to suggest a question that can be answered based on the introductory section of the page, using the following prompt:

Your task is to formulate a question from given context satisfying the rules given below:

1. The question should be fully answered from the given context.

2. The question should be framed from a part that contains non-trivial information.

3. The answer should not contain any

links.

4. The question should be of moderate difficulty.

6. Do not use phrases that ’provided con  
text’, etc in the question   
context:

5. The question must be reasonable and must be understood and responded to by humans.

We also used ChatGPT to answer the generated question, when given the corresponding introductory section as context, using the following prompt:

Answer the question using the informa-

tion from the given context.

question: [question]

context: [context]

All questions were annotated along the three considered quality dimensions by two annotators. Both annotators were fluent in English and were given clear instructions about the meaning of the three considered quality dimensions. For faithfulness and context relevance, the two annotators agreed in around 95% of cases. For answer relevance, they agreed in around 90% of the cases. Disagreements were resolved after a discussion between the annotators.

Faithfulness To obtain human judgements about faithfulness, we first used ChatGPT to answer the question without access to any additional context. We then asked the annotators to judge which of the two answers was the most faithful (i.e. the standard one or the one generated without context), given the question and corresponding Wikipedia page.

Answer relevance We first used ChatGPT to obtain candidate answers with lower answer relevance, using the following prompt:

Answer the given question in an incom-

plete manner.

question: [question]

We then asked human annotators to compare this answer, and indicate which of the two answers had the highest answer relevance.

Context relevance To measure this aspect, we first added additional sentences to the context by scraping back-links to the corresponding Wikipedia page. In this way, we were able to add information to the context that was related but less relevant for