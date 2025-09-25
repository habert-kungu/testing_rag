# RAG Tool

This is a command-line tool for performing Retrieval-Augmented Generation (RAG) on your documents. You can ingest JSON documents into a MongoDB vector store and then ask questions to get answers based on the content of those documents.

## Prerequisites

- Docker
- Node.js (for local development)
- A MongoDB Atlas cluster
- A Gemini API key

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/habert-kungu/RAG.git
    cd RAG
    ```

2.  **Create a `.env` file:**
    Create a `.env` file in the root of the project and add your MongoDB URI and Gemini API key:
    ```
    MONGODB_URI=your_mongodb_uri
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Docker Usage

1.  **Build the Docker image:**
    ```bash
    docker build -t rag-tool .
    ```

2.  **Run the Docker container:**
    To run a command, you can use `docker run`. Make sure to pass the `.env` file to the container.

    **Ingest a document:**
    Place your JSON file in a local directory (e.g., `./data`) and mount it to the container.

    ```bash
    docker run --rm -v $(pwd)/data:/usr/src/app/data --env-file .env rag-tool ingest ./data/your_document.json
    ```

    **Query your documents:**
    ```bash
    docker run --rm --env-file .env rag-tool query "Your question here"
    ```

## Local Development

1.  **Install dependencies:**
    ```bash
    npm install
    ```

2.  **Run the tool:**
    You can use `ts-node` to run the tool directly.

    **Ingest a document:**
    ```bash
    npx ts-node src/index.ts ingest src/sample_files/data.json
    ```

    **Query your documents:**
    ```bash
    npx ts-node src/index.ts query "What is MongoDB?"
    ```
