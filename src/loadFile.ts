import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { JSONLoader } from "langchain/document_loaders/fs/json";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { config } from "./config";

export async function ingestDocument(filePath: string) {
  const loader = new JSONLoader(filePath);

  const pages = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.textSplitter.chunkSize,
    chunkOverlap: config.textSplitter.chunkOverlap,
  });

  const splitDocs = await textSplitter.splitDocuments(pages);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: config.llm.apiKey,
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  console.log(`Successfully ingested ${filePath}`);
  return vectorStore;
}
