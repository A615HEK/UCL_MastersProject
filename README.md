<!-- # Structured outputs from unstructured documents -->
# Introduction
The project aims to provide structured outputs from unstructured and messy real-world documents such as FDA approval documents.
<br>Source grounding is established by the use of a RAG model that provides only relevant parts of the document to be processed to the generating LLM.
<br>To ensure that the generated answers are in sync with the information in the FDA approval documents, an LLM-as-a-Judge (LLMaJ) approach is undertaken to evaluate the answers/information that are extracted/generated in each cycle of the loop.
<br>All systems were locally run using Ollama.

# System Overview
+ Generating model :- llama3.2:1b
+ Judging model :- llama3.1:8b
+ Embedding model :- all-MiniLm-L6-v2
+ Vector store :- FAISS (Facebook AI Similarity Search)

# Results
- Reduced hallucinations by \~ 90%
- Sped up answer generation time by 30%
- Sped up judging time by around 3x
- Final Human-LLM agreement score: 85-90% (Comparable to SoTA LLMs)

# Acknowledgements
I thank my mentors Dr. Sahan Bulathwela (Lecturer, UCL - Internal supervisor) and Dr. Kanwal Bhatia (CEO, Aival - Industry supervisor) for their valuable inputs and guidance throughout the duration of the project. I would also like to thank my parents, sister, and my friends who were with me throughout my time at UCL, cheering me on every step along the journey.