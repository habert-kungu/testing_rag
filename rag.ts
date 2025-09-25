import { MongoClient, Collection } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import {
  GoogleGenerativeAIEmbeddings,
  ChatGoogleGenerativeAI,
} from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "langchain/document";
import * as fs from "fs";
import * as readline from "readline";
import { config } from "./config.js";
import cosineSimilarity from "compute-cosine-similarity";

async function createInMemoryVectorStore(
  documents: any[],
  embeddings: GoogleGenerativeAIEmbeddings
) {
  const vectorStore = [];

  for (const doc of documents) {
    const content = `Question: ${doc.question}\nAnswer: ${doc.answer}`;
    const embedding = await embeddings.embedQuery(content);
    vectorStore.push({
      embedding,
      document: doc,
    });
  }

  return vectorStore;
}

async function findMostRelevantFAQ(
  query: string,
  vectorStore: any[],
  embeddings: GoogleGenerativeAIEmbeddings
) {
  const queryEmbedding = await embeddings.embedQuery(query);
  let maxSimilarity = -Infinity;
  let mostRelevantFAQ = null;

  for (const item of vectorStore) {
    const itemEmbedding = await item.embedding;
    const similarity = cosineSimilarity(queryEmbedding, itemEmbedding);
    if (similarity !== null && similarity > maxSimilarity) {
      maxSimilarity = similarity;
      mostRelevantFAQ = item.document;
    }
  }

  return mostRelevantFAQ;
}

async function answerQuestionInMemory(
  query: string,
  vectorStore: any[],
  embeddings: GoogleGenerativeAIEmbeddings
) {
  const mostRelevantFAQ = await findMostRelevantFAQ(
    query,
    vectorStore,
    embeddings
  );

  if (!mostRelevantFAQ) {
    return "I'm sorry, I couldn't find a relevant FAQ.";
  }

  const llm = new ChatGoogleGenerativeAI({
    apiKey: config.llm.apiKey,
    temperature: 0,
    model: config.llm.model,
  });

  const prompt = PromptTemplate.fromTemplate(
    `Use the following FAQ to answer the question:
        FAQ: {faq}
        Question: {question}
        Answer:`
  );

  const chain = RunnableSequence.from([
    {
      faq: await mostRelevantFAQ.answer,
      question: async () => query,
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  const result = await chain.invoke({
    faq: await mostRelevantFAQ.answer,
    question: query,
  });
  return { faq: mostRelevantFAQ, answer: result };
}

async function main() {
  const fileContent = fs.readFileSync("data.json", "utf-8");
  const documents = fileContent
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line));

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: config.llm.apiKey,
  });
  const vectorStore = await createInMemoryVectorStore(documents, embeddings);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "Ask a question> ",
  });

  console.log('You can now ask questions. Type "exit" to quit.');
  rl.prompt();

  for await (const line of rl) {
    if (line.toLowerCase() === "exit") {
      break;
    }
    const response = await answerQuestionInMemory(
      line,
      vectorStore,
      embeddings
    );
    if (typeof response === "string") {
      console.log(response);
    } else {
      console.log("FAQ:", response.faq);
      console.log("Answer:", response.answer);
    }
    rl.prompt();
  }
}

main().catch(console.error);
