import * as dotenv from "dotenv";
dotenv.config();

export const config = {
  mongodb: {
    uri: process.env.MONGODB_URI!,
    dbName: "test",
    collectionName: "embeddings",
  },
  llm: {
    apiKey: process.env.GOOGLE_API_KEY || "",
    model: "gemini-1.5-flash",
  },
  vectorStore: {
    indexName: "default",
  },
  ingestion: {
    filePath: "/home/habert/test/data.json",
  },
};
