# Conversational RAG Assistant

## Overview

This project implements a Conversational RAG (Retrieval-Augmented Generation) Assistant with:

* Conversation Memory
* History-Aware Retrieval
* Tool Calling
* Routing Logic

## Technologies Used

* Python
* LangChain
* FAISS
* HuggingFace Embeddings
* Ollama (Llama3)

## Features

* Maintains chat history for follow-up questions
* Rewrites follow-up queries into standalone questions
* Retrieves relevant information from a knowledge base
* Uses an external tool to fetch the current time
* Routes queries between RAG and tool execution

## Project Structure

conversational-rag-assistant/

* app.py
* tools.py
* knowledge_base.txt
* requirements.txt
* README.md

## Run the Project

Install dependencies:

pip install -r requirements.txt

Run the application:

python3 app.py

## Example Questions

* What is RAG?
* Can you explain it more?
* What is the current time?
