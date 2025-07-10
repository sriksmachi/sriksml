# 🚀 SriksML: Advanced Machine Learning & AI Experiments

Welcome to **SriksML** – a comprehensive repository of hands-on, production-inspired Jupyter notebooks and code samples for modern machine learning, deep learning, and AI workflows. This repository is designed for practitioners, researchers, and enthusiasts who want to go beyond the basics and explore real-world ML/AI systems, from classical models to cutting-edge generative AI and reinforcement learning.

---

## 📂 Repository Structure & High-Level Objectives

- **azureml/**: End-to-end Azure ML workflows, agentic services, and cloud-scale model management.
- **classical-ml-labs/**: Core machine learning concepts, algorithms, and practical data science labs.
- **genai-labs/**: Deep dives into generative AI, language models, transformers, and advanced NLP.
- **images/**: Visual assets, architecture diagrams, and illustrations for ML/AI concepts.
- **projects/**: Real-world, domain-driven ML projects, including causal learning on telemetry data.
- **reinforcement-learning-labs/**: Custom environments and implementations of RL algorithms.

---

## 🔷 azureml/
**Objective:** Enable scalable, reproducible ML on Azure, including agentic workflows and model finetuning.

- **001-azure_agentic_service_autogen.ipynb**: Build and orchestrate AI agents using Azure Agentic Service, Bing, and LLMs.  
  *Key Learnings:* Multi-agent orchestration, workflow automation, summarization pipelines.
- **finetuning_hf_models/**: End-to-end HuggingFace model finetuning on Azure ML, with lineage tracking and deployment.
- **azure-ai-mcp/**: Custom clients and server for Azure AI Model Context Protocol (MCP).
- **data/**: Use cases and supporting datasets for Azure ML experiments.

---

## 📊 classical-ml-labs/
**Objective:** Master foundational ML algorithms and data science workflows.

- **001-credit-card-fraud-detection.ipynb**: End-to-end fraud detection using supervised learning and feature engineering.
- **002-pca.ipynb**: Principal Component Analysis for dimensionality reduction and visualization.
- **003-recommendation-system-with-sentiment.ipynb**: Hybrid recommender systems integrating sentiment analysis.
- **004-recommendation-system.ipynb**: Collaborative filtering and content-based recommendation.
- **005-ResamplingApproaches.ipynb**: Advanced resampling techniques for model validation and robustness.

---

## 🤖 genai-labs/
**Objective:** Explore the latest in generative AI, LLMs, and advanced NLP.

- **001-building-BERT-from-scratch.ipynb**: Implement BERT from the ground up, including tokenization and masked language modeling.
- **002-fine_tuning_llms_rlhf.ipynb**: Fine-tune LLMs using Reinforcement Learning from Human Feedback (RLHF).
- **003-finetuning_flant5.ipynb**: Task-specific fine-tuning of FLAN-T5 for summarization, Q&A, and translation.
- **004-graphrag_ds.ipynb**: Graph-based Retrieval-Augmented Generation (RAG) with Neo4j and Azure OpenAI.
- **005-knowledge_distillation_llm.ipynb**: Compress LLMs via knowledge distillation for efficient deployment.
- **006-langchain_cypherchain_humaninloop.ipynb**: Human-in-the-loop graph RAG with LangChain and Neo4j.
- **007-optimized_vector_embeddings.ipynb**: Techniques for optimizing vector embeddings (WIP).
- **008-rnn-glove-embeddings.ipynb**: RNNs for text generation with GloVe embeddings.
- **009-rnn-trained-embeddings.ipynb**: RNNs with self-trained embeddings for domain-specific tasks.
- **010-transformers_101_nmt_eng_hin.ipynb**: Transformers for English-Hindi neural machine translation.
- **011-transformers_101.ipynb**: Foundational transformer concepts and implementations.
- **012-variational-auto-encoders.ipynb**: End-to-end VAEs for generative modeling and creative vector math.

---

## 🏗️ projects/
**Objective:** Apply ML/AI to real-world, domain-specific problems.

- **causallearning_on_telemetrydata/**:  
  - **data_prep.ipynb**: Data wrangling and preprocessing for telemetry.
  - **rca_gcm.ipynb**: Root cause analysis using graphical causal models.
  - **graph_builder.py**: Automated graph construction from telemetry.
  - **kql_queries.yaml**: Kusto queries for Azure Monitor integration.
  - *Key Learnings:* Causal inference, time series analysis, graph-based diagnostics, and integration with cloud telemetry.

---

## 🕹️ reinforcement-learning-labs/
**Objective:** Build and experiment with RL algorithms and environments.

- **001-dqn_custom_env.ipynb**: Deep Q-Networks in custom simulation environments.
- **002-ppo.ipynb**: Proximal Policy Optimization for continuous control.
- **003-reinforce.ipynb**: Policy gradient methods from scratch.
- **contosocabs_env.py**: Custom RL environment for cab dispatch simulation.

---

## 🌟 Why Explore SriksML?

- 📚 **Comprehensive Coverage:** From classical ML to the latest in generative AI and RL.
- 🧑‍💻 **Production-Ready Patterns:** Real-world code, not just toy examples.
- 🧩 **Modular & Reusable:** Each notebook is self-contained and ready for adaptation.
- 🏆 **Cutting-Edge Techniques:** RLHF, RAG, agentic workflows, and more.
- 🖼️ **Rich Visuals:** Diagrams and images to clarify complex concepts.

---

**Dive in, experiment, and accelerate your ML/AI journey with SriksML!**  
*Star the repo, open issues, and contribute to push the boundaries of applied machine learning together.*
