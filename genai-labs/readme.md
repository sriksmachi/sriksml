# Introduction

This folder contains Jupyter notebooks that explore a wide range of modern language modeling and generative AI techniques. The collection covers foundational models, advanced fine-tuning, retrieval-augmented generation, knowledge distillation, graph-based approaches, and practical applications with human-in-the-loop and Azure AI services. Each notebook is designed to be self-contained and provides hands-on code, explanations, and references.

## Notebooks Index

| Notebook | Short Description | Key Learnings, Objectives & Summary |
|---|---|---|
| [001-building-BERT-from-scratch.ipynb](001-building-BERT-from-scratch.ipynb) | Build and train a BERT model from scratch. | Learn the architecture and training process of BERT, including tokenization, embedding, and masked language modeling. Gain hands-on experience with PyTorch and NLP preprocessing. |
| [002-fine_tuning_llms_rlhf.ipynb](002-fine_tuning_llms_rlhf.ipynb) | Fine-tune large language models using RLHF. | Understand Reinforcement Learning from Human Feedback (RLHF) for LLMs. Practice fine-tuning with HuggingFace, evaluate models, and explore reward modeling. |
| [003-finetuning_flant5.ipynb](003-finetuning_flant5.ipynb) | Fine-tune the FLAN-T5 model for multiple tasks. | Explore task-specific fine-tuning (summarization, Q&A, translation) on FLAN-T5. Learn about model architecture, parameter inspection, and evaluation. |
| [004-graphrag_ds.ipynb](004-graphrag_ds.ipynb) | Graph-based Retrieval-Augmented Generation (RAG) with Neo4j. | Implement RAG using knowledge graphs, Neo4j, and Azure OpenAI. Learn about graph-based context retrieval, clustering, and interpretability in RAG systems. |
| [005-knowledge_distillation_llm.ipynb](005-knowledge_distillation_llm.ipynb) | Knowledge distillation for LLMs. | Apply knowledge distillation to compress large models into smaller ones. Learn about teacher-student training, soft labels, and practical distillation for Q&A tasks. |
| [006-langchain_cypherchain_humaninloop.ipynb](006-langchain_cypherchain_humaninloop.ipynb) | Graph RAG with human-in-the-loop using LangChain and Neo4j. | Dissect Graph RAG pipelines, integrate human feedback for query refinement, and use LangChain with Azure OpenAI for robust graph-based QA. |
| [007-optimized_vector_embeddings.ipynb](007-optimized_vector_embeddings.ipynb) | Optimizing vector embeddings for retrieval tasks. | Explore techniques for improving embedding quality and retrieval performance. (Notebook is a placeholder or under development.) |
| [008-rnn-glove-embeddings.ipynb](008-rnn-glove-embeddings.ipynb) | RNN for text generation with GloVe embeddings. | Compare character-based and word-based RNNs, integrate pre-trained GloVe embeddings, and analyze strengths and limitations of each approach. |
| [009-rnn-trained-embeddings.ipynb](009-rnn-trained-embeddings.ipynb) | RNN for text generation with self-trained embeddings. | Train custom word embeddings, use them in RNNs, and compare with pre-trained embeddings. Understand the benefits of domain-specific representations. |
| [010-transformers_101_nmt_eng_hin.ipynb](010-transformers_101_nmt_eng_hin.ipynb) | Transformers for English-Hindi neural machine translation. | Build a transformer model from scratch, understand attention mechanisms, and apply to sequence-to-sequence translation tasks. |
| [011-transformers_101.ipynb](011-transformers_101.ipynb) | Transformers 101: foundational concepts and implementation. | Learn the basics of transformer models, attention, and encoder-decoder architectures. Follow a step-by-step implementation for language tasks. |
| [012-variational-auto-encoders.ipynb](012-variational-auto-encoders.ipynb) | End-to-end Variational Autoencoders (VAE) for generative modeling. | Build and train VAEs, understand latent space, reparameterization trick, and generate new samples. Apply vector math for creative manipulations. |

---

Feel free to explore the notebooks to deepen your understanding of language modeling, generative AI, and practical applications!
