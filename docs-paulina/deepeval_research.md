# DeepEval

<aside>
📌

**Metric:** standard of measurement for evaluating performance of an LLM output based on a specific criteria of interest.  

</aside>

<aside>
📌

**G-Eval metric**: custom metric which uses LLM-as-a-judge with chain-of-thought (CoT) to evaluate outputs based on any custom criteria. Requires `input` and `actual_output`. 

</aside>

<aside>
📌

**DAG metric**: custom metric which allows to run evaluations by constructing LLM-powered decision trees for cases where a extremely deterministic score is required. 

</aside>

# Retrieval-Augmented Generation Evaluation

## Retrieval-Augmented Generation (RAG)

- **Retrieval-Augmented Generation** (RAG) is a **technique** used to **enrich LLM outputs** by using **additional relevant information** from an external **knowledge base**. This allows an LLM to generate responses **based on context** beyond the scope of its training data.
- The processes of retrieving relevant context, is carried out by the **retriever**, while generating responses based on the **retrieval context**, is carried out by the **generator**. Together, the retriever and generator forms your **RAG pipeline.**
- The **retrieval context** (ie. a list of text chunks) is what the retriever retrieves, while the **LLM output** is what the generator generates.
- RAG evaluation focuses on **evaluating** the **retriever** and **generator** in your RAG pipeline **separately**. This also allows for easier debugging and to **pinpoint issues** on a **component level.**
- Retrieval and generation steps are **influenced** by your **choice of hyperparameters.** Hyperparameters include things like the **embedding model** to use for retrieval, the **number of nodes to retrieve** (we'll just be referring to just as **"top-K"** from here onwards), **LLM temperature**, **prompt template**, etc.

### Retrieval

The retrieval step typically involves:

1. **Vectorizing the initial input into an embedding**, using an embedding model of your choice
2. **Performing a vector search** (by using the previously embedded input) on the vector store that contains your vectorized knowledge base, to retrieve the top-K most "similar" vectorized text chunks in your vector store.
3. **Re-rank the retrieved nodes**. The initial ranking provided by the vector search might not always align perfectly with the specific relevance for your specific use-case.

As you've noticed, there are **quite a few hyperparameters** such as the choice of embedding model, top-K, etc. that **need tuning**. Here are some **questions** RAG **evaluation aims to solve** in the retrieval step:

<aside>
❓

- **Does the embedding model you're using capture domain-specific nuances?** (If you're working on a medical use case, a generic embedding model might not provide the expected vector search results.)
- **Does your re-ranker model ranks the retrieved nodes in the "correct" order?**
- **Are you retrieving the right amount of information?** This is influenced by text chunk size, and top-K number.
</aside>

### Generation

The generation step, following the retrieval step, typically involves:

1. **Constructing a prompt** based on the initial input and the previous vector-fetched retrieval context.
2. **Providing this prompt to your LLM.** This yields the final augmented output.

Although this step is typically more straightforward there are still some questions evaluation can answer at this level:

<aside>
❓

- **Can you use a smaller, faster, cheaper LLM?** **Would finetuning help?** This often involves exploring open-source alternatives like LLaMA-2, Mistral 7B, and fine-tuning your own versions of it.
- **Would a higher temperature give better results?**
- **How does changing the prompt template affect output quality?** This is where most LLM practitioners spend most time on.
</aside>

Usually you'll find yourself starting with a state-of-the-art model, and moving to smaller, or even fine-tuned, models where possible, and it is the many different versions of prompt template where LLM practitioners lose control of.

## RAG Metrics

- LLM-as-a-judge to measure RAG quality
- Self-explaining metrics which output a reason for every metric score

<aside>
🚩

The `input` and `actual_output` are required to create an `LLMTestCase` (and hence required by all metrics) even though they might not be used for metric calculation.

</aside>

## Retrieval Metrics

<aside>
🚩

You should aim to use all three metrics in conjunction for comprehensive evaluation results.

</aside>

A **combination of these three metrics are needed** because, you want to make sure the retriever is able to retrieve just the right amount of information, in the right order. RAG evaluation in the retrieval step ensures you are feeding **clean data** to your generator.

## Contextual Relevancy

<aside>
✍️

[`ContextualRelevancyMetric`](https://www.deepeval.com/docs/metrics-contextual-relevancy): evaluates whether the **text chunk size** and **top-K** of your retriever is able to retrieve information without much irrelevancies.

</aside>

## Contextual Precision

<aside>
✍️

[`ContextualPrecisionMetric`](https://www.deepeval.com/docs/metrics-contextual-precision): evaluates whether the **reranker** in your retriever ranks more relevant nodes in your retrieval context higher than irrelevant ones.

</aside>

- How relevant `retrieval_context` is to `input`.
- A higher contextual precision score represents a greater ability of the retrieval system to correctly rank relevant nodes higher in the `retrieval_context`.
- **How it is calculated?**
    
    $$
     \frac{1}{\text{Nº Relevant Nodes}} \sum_{k=1}^n \left(\frac{\text{Nº Relevant Nodes Up to Position }k}{k} \times r_k\right)
    $$
    
    First it uses an LLM to determine for each node in the `retrieval_context` whether it is relevant to the `input` based on information in the `expected_output`, before calculating the **weighted cumulative precision** as the contextual precision score.
    
    The weighted cumulative precision (WCP) is used because it:
    
    - **Emphasizes on Top Results**: WCP places a stronger emphasis on the relevance of top-ranked results. This emphasis is important because LLMs tend to give more attention to earlier nodes in the `retrieval_context` (which may cause downstream hallucination if nodes are ranked incorrectly).
    - **Rewards Relevant Ordering**: WCP can handle varying degrees of relevance (e.g., "highly relevant", "somewhat relevant", "not relevant"). This is in contrast to metrics like precision, which treats all retrieved nodes as equally important.

## Contextual Recall

<aside>
✍️

[`ContextualRecallMetric`](https://www.deepeval.com/docs/metrics-contextual-recall): evaluates whether the **embedding model** in your retriever is able to accurately capture and retrieve relevant information based on the context of the input.

</aside>

## Generation Metrics

<aside>
🚩

In reality, the hyperparameters for the generator isn't as clear-cut as hyperparameters in the retriever. *To evaluate generation on customized criteria, you should use the [`GEval`](https://www.deepeval.com/docs/metrics-llm-evals) metric instead, which covers all custom use cases.*

</aside>

<aside>
🚩

Using these scores in conjunction will best align with human expectations of what a good LLM output looks like.

</aside>

### Answer Relevancy

<aside>
✍️

 [`AnswerRelevancyMetric`](https://www.deepeval.com/docs/metrics-answer-relevancy): evaluates whether the **prompt template** in your generator is able to instruct your LLM to output relevant and helpful outputs based on the `retrieval_context`.

</aside>

| How relevant `actual_output` is compared to `input`  |
| --- |
| How relevant  `generated_response` compared to `question` |

### Faithfulness

<aside>
✍️

 [`FaithfulnessMetric`](https://www.deepeval.com/docs/metrics-faithfulness): evaluates whether the **LLM** used in your generator can output information that does not hallucinate **AND** contradict any factual information presented in the `retrieval_context`.

</aside>