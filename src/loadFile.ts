import { MongoClient } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config } from "./config";

export async function ingestDocument(filePath: string) {
  const client = new MongoClient(config.mongodb.uri);
  await client.connect();
  const collection = client
    .db(config.mongodb.dbName)
    .collection(config.mongodb.collectionName);

  const loader = new PDFLoader(filePath);
  const pages = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.textSplitter.chunkSize,
    chunkOverlap: config.textSplitter.chunkOverlap,
  });

  const splitDocs = await textSplitter.splitDocuments(pages);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: config.llm.apiKey,
  });

  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection: collection,
    indexName: config.vectorStore.indexName,
  });

  await vectorStore.addDocuments(splitDocs);

  await client.close();
  console.log(`Successfully ingested ${filePath}`);
}
