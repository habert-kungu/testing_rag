import { MongoClient, Collection } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import * as fs from "fs";
import * as dotenv from 'dotenv';
import * as readline from 'readline';

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
  ingestion: {
      filePath: "/home/habert/test/data.json",
  }
};

async function ingestDocument(collection: Collection, filePath: string) {
    console.log("Starting ingestion from", filePath);
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
    console.log("Ingestion complete.");
}

async function answerQuestion(collection: Collection, question: string) {
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

    const answer = result.content;
    console.log("Answer:", answer);
}

async function main() {
  const client = new MongoClient(config.mongodb.uri);
  await client.connect();
  try {
    const collection = client.db(config.mongodb.dbName).collection(config.mongodb.collectionName);

    const docCount = await collection.countDocuments();
    if (docCount === 0) {
        console.log("No documents found in the collection. Starting ingestion.");
        await ingestDocument(collection, config.ingestion.filePath);
    } else {
        console.log("Found existing documents in the collection. Skipping ingestion.");
    }

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: 'Ask a question> '
    });

    console.log('You can now ask questions. Type "exit" to quit.');
    rl.prompt();

    for await (const line of rl) {
        if (line.toLowerCase() === 'exit') {
            break;
        }
        await answerQuestion(collection, line);
        rl.prompt();
    }

  } finally {
    await client.close();
    console.log("Connection closed.");
  }
}

main().catch(console.error);
