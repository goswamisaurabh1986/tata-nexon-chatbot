Load the retrieval skill.

We now want to add proper citation support to the retrieved chunks.

Update the Retriever in src/retrieval/retriever.py with the following:

1. Each returned chunk must include a `citation_id` field in this format:
   - "{source}:{chunk_index}"   Example: "Tata_Nexon_Brochure.pdf:5"

2. Also include `page_number` in metadata (if available from ingestion).

3. Update the existing tests and add a new test:
   - Verify that every returned chunk contains:
     - "citation_id"
     - "text"
     - "metadata" (with source, section_title, chunk_index)
     - "score"

4. Make sure the dummy chunks in the minimal implementation now return proper citation structure.

After updating, run all tests in tests/retrieval/ and ensure they pass.

At the end, show the updated retrieve() method code.