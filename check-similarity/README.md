# Image Similarity Checker

A lightweight Streamlit front-end that sits on top of the existing CLIP-based
retrieval experiment. It lets you:

- Browse a hard-coded imagery directory (recursively) and pick any PNG/JPG as the query.
- Submit plain-language text prompts.
- Configure the number of neighbours (`top_k`) to retrieve.
- View ranked results with preview thumbnails and cosine similarity scores.

Database host/user credentials live in `app.py` (`DB_CONFIG`). Pass the database
name and imagery directory at launch (`--db-name`, `--image-dir`) or rely on the
built-in defaults.

## Usage

```bash
export STREAMLIT_SERVER_PORT=8501  # optional
pip install -r check-similiarity/requirements.txt
streamlit run check-similiarity/app.py \
  -- --db-name street_images \
     --image-dir data/runtime/universes/neurips3_seg/imagery
```

The app assumes you already ran the image retrieval pipeline so that
`image_embeddings` in PostgreSQL is populated with vectors for the files in
`IMAGE_DIR`.

If Not please run the Experiments/Image_retrieval Setup before doing this.
