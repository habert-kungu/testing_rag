import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { config } from "./config";
import { Document } from "langchain/document";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export async function queryData(query: string, vectorStore: MemoryVectorStore) {
  const retriever = vectorStore.asRetriever({
    k: 5,
  });

  const llm = new ChatGoogleGenerativeAI({
    apiKey: config.llm.apiKey,
    temperature: 0,
    model: config.llm.model,
    maxRetries: 3,
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
    """`
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

  try {
    console.log("Invoking chain with query...");
    const result = await Promise.race([
      chain.invoke(query),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Timeout")), 30000)
      ), // 30 seconds timeout
    ]);
    console.log("Chain invocation complete.");
    return result;
  } catch (error) {
    console.error("Error during query execution:", error);
    throw error;
  }
}
