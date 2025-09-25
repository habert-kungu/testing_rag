import "dotenv/config";
import fs from "fs/promises";
import path from "path";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Document } from "langchain/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// --- 1. Load FAQ data ---
async function loadFaqs(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  const faqs = JSON.parse(raw);
  return faqs
    .filter((f) => f.id && f.question && f.answer)
    .map(
      (f) =>
        new Document({
          pageContent: `${f.question}\n${f.answer}`,
          metadata: { id: f.id, tags: f.tags, updated_at: f.updated_at },
        })
    );
}

// --- 2. Build vector store ---
async function buildVectorStore(docs, apiKey) {
  console.log("Building vector store...");
  try {
    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "embedding-001",
      apiKey,
      maxRetries: 3,
    });

    const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
    console.log("Vector store built successfully!");
    return store;
  } catch (error) {
    console.error("Error building vector store:", error.message);
    throw error;
  }
}

// --- 3. Query pipeline ---
async function queryFaq(store, query, apiKey) {
  console.log("Starting query process...");
  try {
    const retriever = store.asRetriever(2);
    console.log("Getting relevant documents...");

    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-pro",
      apiKey,
      temperature: 0.2,
      maxRetries: 3,
    });

    // Get top matches
    const docs = await retriever.getRelevantDocuments(query);
    console.log("Found relevant documents:", docs.length);

    const context = docs
      .map(
        (d) =>
          `Q: ${d.pageContent.split("\n")[0]}\nA: ${
            d.pageContent.split("\n")[1]
          }`
      )
      .join("\n\n");

    const prompt = `
You are an HR assistant. A user asked: "${query}".

Here are some relevant FAQs:
${context}

Give a clear, concise, and helpful answer grounded in the FAQ.
If not sure, say you don't know.
`;

    console.log("Generating response...");
    const resp = await llm.invoke(prompt);
    return resp.content;
  } catch (error) {
    console.error("Error in query process:", error.message);
    throw error;
  }

  const context = docs
    .map(
      (d) =>
        `Q: ${d.pageContent.split("\n")[0]}\nA: ${d.pageContent.split("\n")[1]}`
    )
    .join("\n\n");

  const prompt = `
You are an HR assistant. A user asked: "${query}".

Here are some relevant FAQs:
${context}

Give a clear, concise, and helpful answer grounded in the FAQ.
If not sure, say you don’t know.
`;

  const resp = await llm.invoke(prompt);
  return resp.content;
}

// --- 4. Run ---
async function main() {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error("Set GEMINI_API_KEY env var");

    const faqFile = path.resolve("./faqs.json");
    const docs = await loadFaqs(faqFile);

    console.log(`Loaded ${docs.length} FAQ entries.`);

    const store = await buildVectorStore(docs, apiKey);

    // Get query from command line arguments
    const query = process.argv[2];

    if (!query) {
      console.log("Please provide a query as an argument.");
      console.log('Usage: node src/rag.js "your question here"');
      process.exit(1);
    }

    const answer = await queryFaq(store, query, apiKey);

    console.log("\nQuestion:", query);
    console.log("\nAnswer:", answer);
  } catch (error) {
    console.error("\nError:", error.message);
    process.exit(1);
  }
}

main();
