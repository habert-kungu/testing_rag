import { MongoClient } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as fs from "fs";

// Add a basic config structure
const config = {
  mongodb: {
    uri: process.env.MONGODB_URI || "mongodb://localhost:27017",
    dbName: "test",
    collectionName: "embeddings",
  },
  llm: {
    apiKey: process.env.GOOGLE_API_KEY || "",
  },
  vectorStore: {
    indexName: "default",
  },
};

export async function ingestDocument(filePath: string) {
  const client = new MongoClient(config.mongodb.uri);
  await client.connect();
  const collection = client
    .db(config.mongodb.dbName)
    .collection(config.mongodb.collectionName);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: config.llm.apiKey,
  });

  const vectorstore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection,
    indexName: config.vectorStore.indexName,
  });

  const fileContent = fs.readFileSync(filePath, "utf-8");
  const documents = fileContent
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line));

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const docs = await Promise.all(
    documents.map(async (doc) => {
      const content = `Question: ${doc.question}\nAnswer: ${doc.answer}`;
      const chunks = await textSplitter.splitText(content);
      return chunks.map((chunk) => ({
        pageContent: chunk,
        metadata: {
          id: doc.id,
          tags: doc.tags,
          updated_at: doc.updated_at,
        },
      }));
    }),
  );

  const flattenedDocs = docs.flat();
  await vectorstore.addDocuments(flattenedDocs);
  await client.close();
}

ingestDocument("/home/habert/test/data.json")
  .then(() => {
    console.log("Ingestion complete");
  })
  .catch((err) => {
    console.error("Ingestion failed", err);
  });

