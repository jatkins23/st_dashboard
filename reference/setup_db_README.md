## Generate Embeddings
- Images
    `uv run scripts/generate_embeddings.py --db ../st_preprocessing/data/core.ddb --universe nyc  --image-dir /home/jon/code/st_preprocessing/data/imagery/nyc/images --years 2006 2012 2018 2024 --all-pairs --compute-changes --log-file logs/generate_embeddings.log`
- Masks 
    `uv run scripts/generate_embeddings.py --db ../st_preprocessing/data/core.ddb --universe nyc  --image-dir /home/jon/code/st_preprocessing/data/imagery/nyc/masks --years 2006 2012 2018 2024   --all-pairs --compute-changes --log-file logs/generate_embeddings.log`
- SidebySides
    `uv run scripts/generate_embeddings.py --db ../st_preprocessing/data/core.ddb --universe nyc  --image-dir /home/jon/code/st_preprocessing/data/imagery/nyc/sidebysides --years 2006 2012 2018 2024 --all-pairs --compute-changes --log-file logs/generate_embeddings.log`

## G
uv run scripts/build_faiss_indexes.py --db ../st_preprocessing/data/core.ddb --universe nyc --all-years --index-type hnsw --index-dir data2/faiss_indices --cache-dir data2/embedding_cache