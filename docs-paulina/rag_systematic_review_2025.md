Oche, A. J., Folashade, A. G., Ghosal, T., & Biswas, A. (2025). 
*A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: 
Progress, Gaps, and Future Directions*. arXiv. July 28, 2025.

https://arxiv.org/html/2507.18910v1

Summary:

Introduction, Methodology, Foundations of RAG, Year-by-year advancement evaluation, Proprietary Data & Industry Implementation, RAG Systems Evaluation, Challenges, Discussion and Future Direction, Conclusion wih gaps

1. Introduction
Key ideas:
- RAG turns AI more relaible and knowledge-aware
- RAG to mitigate hallucinations with improved factual accuracy 
- RAG for proprietary data
- Search engine and dialogue agent at the same time

Considerations: 
- Handling sensitive information

New perspectives and trends:

2. Methodology: 
- Academic and industry-focused literature
- 2017 to 2025
- Their criteria for the selected papers to review

3. Foundations of RAG
Definition and key concepts 
- Retrieval module plus generation module
- Improve quality of answers in knowledge-intensive tasks
- Formal definition of a RAG model
- Distinction between parametric memory (LLM training) and non-parametric memory (knowledge-base)
- Adresses fixed LLMs knowledge up to training cutoff date
- Typical RAG pipeline: Chunking, Embedding, (Re)ranking and Generation
- Dense embeddings enable semantic matching in contrast with sparse key word search
- Combination of dense retrieval with lightweight filtering to further improve precision
- Reranking extra step provide a refined relevance score with a more accurate but expensive re-ranker model
- RAG as a "live" memory for the LLM
- Including references for traceability
Technical Components
- Retriever Module (Dense Passage Retrieval)
    - Bi-encoder architecture
    - Vector space to compare passage embedding most similar query’s embedding
    - Relevance assessed by similarity score
    - Approximate Nearest Neighbor efficient techniques
    - Output: Z top-ranked     
    - Narrows down evidence the generator will consider
- Generator Module (Conditional Seq2Seq
Model)
    - Provided with question promt and retieved context
    - Multiple ways to provide retrieved context
        - Early fusion - concatenation
        - Late fusion - no la pillo
        - Fussion Mechanisms and Answer Aggregation - area of research to improve answer completeness and correctness
    - Training and optimization 
        - Finetuning QA knowledge-intensive dialogue dataset
        - Tune both modules parameters 





