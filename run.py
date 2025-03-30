import sys

from sentence_transformers import SentenceTransformer

from Chunking.fixed_token_chunker import FixedTokenChunker
from retrieval_evaluation_pipeline import RetrievalEvaluationPipeline


CHUNKERS = {
    "FixedTokenChunker": lambda chunk_size: FixedTokenChunker(chunk_size=chunk_size)
}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <chunker> <embedding_function> <no_retrieved_chunks>")
        sys.exit(1)

    chunk_size = 800
    chunker_name = sys.argv[1]

    chunker_constructor = CHUNKERS.get(chunker_name)
    if chunker_constructor:
        chunker = chunker_constructor(chunk_size)
    else:
        print(f"Error: Unknown chunker '{chunker_name}'. Available options: {', '.join(CHUNKERS.keys())}")
        sys.exit(1)

    try:
        model_name = sys.argv[2]
        model = SentenceTransformer(f"sentence-transformers/{model_name}")
    except Exception as e:
        print(f"Error: Could not load embedding function '{model_name}'. Please check the model name.")
        sys.exit(1)

    try:
        no_retrieved_chunks = int(sys.argv[3])
        if no_retrieved_chunks < 1:
            raise ValueError
    except ValueError:
        print("Number of retrieved chunks must be a positive integer")
        sys.exit(1)

    pipeline = RetrievalEvaluationPipeline(chunker, no_retrieved_chunks, model)
    precision_score, recall_score = pipeline.evaluate()
    print(f"Chunk size: {chunk_size}, Number of retrieved chunks: {no_retrieved_chunks}, Precision score: {precision_score}, Recall score: {recall_score}")
