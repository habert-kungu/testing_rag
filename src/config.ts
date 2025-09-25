import "dotenv/config";

export const config = {
  mongodb: {
    uri: process.env.MONGODB_URI!,
    dbName: "book_mongodb_chunks",
    collectionName: "chunked_data",
  },
  llm: {
    apiKey: process.env.GEMINI_API_KEY!,
    model: "gemini-1.5-flash",
  },
  textSplitter: {
    chunkSize: 500,
    chunkOverlap: 150,
  },
  vectorStore: {
    indexName: "default",
  },
};
