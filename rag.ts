import { MongoClient } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import * as fs from "fs";
import * as dotenv from 'dotenv';

dotenv.config();

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
  try {
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
    console.log("Ingestion complete");
  } finally {
    await client.close();
  }
}

async function answerQuestion(question: string) {
  const client = new MongoClient(config.mongodb.uri);
  await client.connect();
  try {
    const collection = client.db(config.mongodb.dbName).collection(config.mongodb.collectionName);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: config.llm.apiKey,
    });

    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
      collection,
      indexName: config.vectorStore.indexName,
    });

    const retriever = vectorStore.asRetriever();

    const prompt = ChatPromptTemplate.fromTemplate(`
      Answer the following question based only on the provided context:
      <context>
      {context}
      </context>
      Question: {question}
    `);

    const model = new ChatGoogleGenerativeAI({
        apiKey: config.llm.apiKey,
        modelName: "gemini-pro",
        temperature: 0.3,
    });

    const chain = RunnableSequence.from([
      {
        context: retriever.pipe(formatDocumentsAsString),
        question: (input) => input.question,
      },
      prompt,
      model,
    ]);

    const result = await chain.invoke({
        question,
    });

    // The result from ChatGoogleGenerativeAI is a message object. We need to access the content.
    // The exact structure might vary, so let's check for common content properties.
    const answer = result.content;

    console.log("Answer:", answer);

  } finally {
    await client.close();
  }
}


async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === "ingest") {
    const filePath = args[1];
    if (!filePath) {
      console.error("Please provide a file path for ingestion.");
      return;
    }
    await ingestDocument(filePath);
  } else if (command === "ask") {
    const question = args.slice(1).join(" ");
    if (!question) {
      console.error("Please provide a question.");
      return;
    }
    await answerQuestion(question);
  } else {
    console.log("Unknown command. Available commands: ingest <filePath>, ask <question>");
  }
}

main().catch(console.error);