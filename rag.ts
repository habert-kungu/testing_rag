import { MongoClient, Collection } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "langchain/document";
import * as fs from "fs";
import * as readline from 'readline';
import { config } from "./config.js";

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

async function answerQuestion(collection: Collection, query: string) {
    const vectorStore = new MongoDBAtlasVectorSearch(
        new GoogleGenerativeAIEmbeddings({ apiKey: config.llm.apiKey }),
        {
          collection: collection,
          indexName: config.vectorStore.indexName,
        },
      );

      const retriever = vectorStore.asRetriever({
        k: 5,
      });

      const llm = new ChatGoogleGenerativeAI({
        apiKey: config.llm.apiKey,
        temperature: 0,
        model: config.llm.model,
      });

      const prompt = PromptTemplate.fromTemplate(
        `Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Do not answer the question if there is no given context.
        Do not answer the question if it is not related to the context.
        Do not give recommendations to anything other than MongoDB.
        Context:
        {context}
        Question: {question}
        """`,
      );

      const formatDocs = (docs: Document[]) => {
        return docs.map((doc) => doc.pageContent).join("\n\n");
      };

      const chain = RunnableSequence.from([
        {
          context: retriever.pipe(formatDocs),
          question: (input) => input,
        },
        prompt,
        llm,
        new StringOutputParser(),
      ]);

      const result = await chain.invoke(query);
      console.log("Answer:", result);
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
