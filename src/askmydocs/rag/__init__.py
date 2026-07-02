"""Retrieval-Augmented Generation (RAG) pipeline for AskMyDocs.

This sub-package groups the building blocks of the document ingestion
and retrieval chain:

- ``loader`` -- extract raw text from source documents (PDF, etc.).
- ``chunker`` -- split long text into fixed-size overlapping chunks
  ready for embedding.
"""
