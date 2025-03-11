# Lab3
Project Overview
This project utilizes the LLaMA 3-70B-8192 model through the Groq API to perform text classification and structured text analysis. It supports various NLP tasks, including text completion, classification with confidence scores, streaming responses, and comparing different prompt strategies for classification accuracy.

Features
Text Completion: Generates text completions based on user input.
Structured Analysis: Formats responses with clearly defined sections for better readability.
Text Classification with Confidence: Assigns categories to text and provides a confidence score.
Streaming API Calls: Streams responses until a specific marker is detected.
Prompt Strategy Comparison: Evaluates different prompting techniques (basic, structured, few-shot) to determine classification accuracy.
API Calls
Uses the Groq API for generating completions and classifications.
Implements error handling for rate limits and API failures.
Supports retry mechanisms for robustness.
Purpose
The project explores how different prompting techniques influence the performance of large language models in classification tasks. By comparing strategies, it helps determine the most effective approach for specific NLP tasks, making it useful for AI research and application development.
