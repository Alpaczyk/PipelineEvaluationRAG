import json
import os.path
import sys

import numpy as np
import pandas as pd
import chromadb

from Chunking.fixed_token_chunker import FixedTokenChunker
from Evaluation.evaluation import Evaluator
from utils import rigorous_document_search
from sentence_transformers import SentenceTransformer


class RetrievalEvaluationPipeline:
    def __init__(self, chunker, no_retrieved_chunks, model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'), questions_csv_path="./Data/questions_df.csv", corpus_path="./Data/state_of_the_union.md"):

        self.chunker = chunker
        self.model = model
        self.no_retrieved_chunks = no_retrieved_chunks
        self.questions_csv_path = questions_csv_path
        self.corpus_path = corpus_path
        self.chroma_client = chromadb.Client()

        self._load_questions_df()
        self._get_chunks_and_metadata()
        self._create_collection()
        self._chunks_to_collection()
        self._create_query_embeddings()
        self.evaluator = Evaluator(self.questions_df)


    def _load_questions_df(self):
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
            self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        else:
            self.questions_df = pd.DataFrame(columns=['question', 'references'])


    def _get_chunks_and_metadata(self):
        documents = []
        metadatas = []

        with open(self.corpus_path, 'r', encoding='utf-8') as file:
            corpus = file.read()
        current_documents = self.chunker.split_text(text=corpus)
        current_metadatas = []
        for document in current_documents:
            try:
                _, start_idx, end_idx = rigorous_document_search(corpus, document)
            except:
                raise Exception(f"Error in finding {document}")
            current_metadatas.append({"start_idx": start_idx, "end_idx": end_idx})
        documents.extend(current_documents)
        metadatas.extend(current_metadatas)
        self.docs = documents
        self.metas = metadatas

    def _create_collection(self):
        collection_name = "chunk_collection"

        try:
            self.chroma_client.get_collection(collection_name)
            self.chroma_client.delete_collection(collection_name)
        except Exception as e:
            pass

        self.chroma_chunk_collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:search_ef": 50}
        )

    def _chunks_to_collection(self):
        batch_size = 100
        for i in range(0, len(self.docs), batch_size):
            batch_docs = self.docs[i:i + batch_size]
            batch_metas = self.metas[i:i+batch_size]
            batch_ids = [str(i) for i in range(i, i+len(batch_docs))]
            batch_embedding = self._get_embeddings(batch_docs)
            self.chroma_chunk_collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                embeddings= batch_embedding
            )

    def _create_query_embeddings(self):
        self.query_embeddings = []
        for x in self.questions_df['question']:
            self.query_embeddings.append(self._get_embeddings(x))


    def _get_embeddings(self, batch_text):
        return self.model.encode(batch_text)

    def _retrieve_chunks(self):
        return self.chroma_chunk_collection.query(query_embeddings=list(self.query_embeddings), n_results=self.no_retrieved_chunks)

    def evaluate(self):
        retrieved_chunks = self._retrieve_chunks()
        precision_scores, recall_scores = self.evaluator.precision_recall_scores(retrieved_chunks['metadatas'])
        return float(np.mean(precision_scores)), float(np.mean(recall_scores))


















