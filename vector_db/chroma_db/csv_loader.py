"""
Multiprocess CSV -> ChromaDB loader
- Streaming via pandas chunksize
- Auto-detect numeric & datetime columns
- Exclude numeric/datetime from semantic text
- Compute embeddings in worker processes (ProcessPoolExecutor)
- Main process performs ChromaDB inserts with returned embeddings
"""

import pandas as pd
import dateutil.parser
from textwrap import wrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from chromadb import Client
import math
from tqdm import tqdm

# ===== CONFIG =====
CSV_FILE = "data.csv"
READ_CHUNK_ROWS = 1000       # rows read per pandas chunk
TEXT_CHUNK_SIZE = 500        # chars per text chunk for embeddings
WORKER_BATCH_ROWS = 500      # how many dataframe rows each worker receives (internally will be <= READ_CHUNK_ROWS)
INSERT_BATCH_SIZE = 2000     # how many embedding chunks to insert per collection.add call
MAX_WORKERS = 4              # processes
COLLECTION_NAME = "csv_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model (change if you prefer)
# ====================

def _is_date(value):
    try:
        dateutil.parser.parse(value)
        return True
    except Exception:
        return False

def detect_column_types(df, sample_n=10):
    numeric_cols = []
    datetime_cols = []
    for col in df.columns:
        # numeric detection
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
            continue
        except Exception:
            pass

        # date detection using small sample
        try:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            sample = non_null.sample(min(sample_n, len(non_null))).astype(str).tolist()
            parsed_ok = sum(1 for v in sample if _is_date(v))
            if parsed_ok / max(1, len(sample)) > 0.8:
                datetime_cols.append(col)
        except Exception:
            pass

    return numeric_cols, datetime_cols

def row_to_text_excluding_cols(row_series, skip_cols):
    # join only non-empty, non-skip columns
    parts = []
    for col, val in row_series.items():
        if col in skip_cols:
            continue
        s = str(val).strip()
        if s:
            parts.append(f"{col}: {s}")
    return "; ".join(parts)

def chunk_text(text, width):
    # simple chunking by characters; keeps semantics intact
    if not text:
        return []
    return wrap(text, width=width)

# Worker function: receives a list of rows (as list-of-dicts) and column type lists.
# Returns lists: ids, metadatas, documents (text), embeddings (vectors)
def worker_embed(batch_rows, batch_idx, numeric_cols, datetime_cols, text_chunk_size, model_name):
    # Each worker loads its own SentenceTransformer model (loaded once per worker process)
    model = SentenceTransformer(model_name)
    skip_cols = set(numeric_cols + datetime_cols)

    ids = []
    metadatas = []
    docs = []

    # Build row-level metadata typed appropriately, and produce text chunks (excluding numeric/dates)
    for local_i, row in enumerate(batch_rows):
        idx = row["_orig_index"]  # original pandas index (stringable)
        # build typed metadata
        meta = {}
        for col, val in row.items():
            if col == "_orig_index":
                continue
            if col in numeric_cols:
                try:
                    meta[col] = float(val) if val != "" else None
                except:
                    meta[col] = None
            elif col in datetime_cols:
                try:
                    meta[col] = dateutil.parser.parse(val).isoformat() if val != "" else None
                except:
                    meta[col] = None
            else:
                meta[col] = val

        # create semantic text excluding numeric/datetime
        # note: row is dict; convert to pandas-like ordering not necessary for semantics
        text = row_to_text_excluding_cols(pd.Series(row).drop(labels=["_orig_index"]), skip_cols)
        chunks = chunk_text(text, text_chunk_size)
        for cidx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            ids.append(f"{batch_idx}_{idx}_{cidx}")
            metadatas.append(meta)
            docs.append(chunk)

    # Compute embeddings for docs (may be empty)
    if docs:
        embeddings = model.encode(docs, show_progress_bar=False, convert_to_numpy=True).tolist()
    else:
        embeddings = []

    # Return small payload to main process
    return {"ids": ids, "metadatas": metadatas, "documents": docs, "embeddings": embeddings}


def main():
    client = Client()
    # create collection WITHOUT built-in embedding function; we'll provide embeddings
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        collection = client.get_collection(name=COLLECTION_NAME)
    else:
        collection = client.create_collection(name=COLLECTION_NAME)

    first_chunk = True
    batch_number = 0
    # We'll accumulate small insertion batches to reduce number of collection.add calls
    pending_ids = []
    pending_metas = []
    pending_docs = []
    pending_embs = []
    numeric_cols = []
    datetime_cols = []

    # Use ProcessPoolExecutor for CPU-bound embedding computation
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        # Read CSV in streaming manner
        for df_chunk in pd.read_csv(CSV_FILE, dtype=str, chunksize=READ_CHUNK_ROWS):
            df_chunk = df_chunk.fillna("").astype(str)
            if first_chunk:
                numeric_cols, datetime_cols = detect_column_types(df_chunk)
                print("Detected numeric columns:", numeric_cols)
                print("Detected datetime columns:", datetime_cols)
                first_chunk = False

            # Split df_chunk into worker-sized batches (list of row dicts)
            nrows = len(df_chunk)
            rows_per_worker = WORKER_BATCH_ROWS
            num_sub_batches = math.ceil(nrows / rows_per_worker)

            for sub_idx in range(num_sub_batches):
                start = sub_idx * rows_per_worker
                end = min((sub_idx + 1) * rows_per_worker, nrows)
                sub_df = df_chunk.iloc[start:end]
                # convert to list of dicts and include original index to create stable IDs
                batch_rows = []
                for orig_idx, row in sub_df.iterrows():
                    row_dict = row.to_dict()
                    row_dict["_orig_index"] = str(orig_idx)
                    batch_rows.append(row_dict)

                # submit to worker
                fut = executor.submit(
                    worker_embed,
                    batch_rows,
                    batch_number,
                    numeric_cols,
                    datetime_cols,
                    TEXT_CHUNK_SIZE,
                    EMBEDDING_MODEL_NAME
                )
                futures.append(fut)
                batch_number += 1

        # collect worker results as they complete
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding workers"):
            res = fut.result()
            # extend pending lists
            pending_ids.extend(res["ids"])
            pending_metas.extend(res["metadatas"])
            pending_docs.extend(res["documents"])
            pending_embs.extend(res["embeddings"])

            # flush to Chroma in INSERT_BATCH_SIZE chunks
            while len(pending_ids) >= INSERT_BATCH_SIZE:
                to_take = INSERT_BATCH_SIZE
                collection.add(
                    ids=pending_ids[:to_take],
                    metadatas=pending_metas[:to_take],
                    documents=pending_docs[:to_take],
                    embeddings=pending_embs[:to_take]
                )
                # drop inserted slice
                pending_ids = pending_ids[to_take:]
                pending_metas = pending_metas[to_take:]
                pending_docs = pending_docs[to_take:]
                pending_embs = pending_embs[to_take:]

    # final flush
    if pending_ids:
        collection.add(
            ids=pending_ids,
            metadatas=pending_metas,
            documents=pending_docs,
            embeddings=pending_embs
        )

    print("✅ All data inserted into ChromaDB.")

if __name__ == "__main__":
    main()
    