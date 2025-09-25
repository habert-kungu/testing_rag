import "dotenv/config";

export const config = {
  llm: {
    apiKey: process.env.GEMINI_API_KEY!,
    model: "gemini-1.5-flash",
  },
  textSplitter: {
    chunkSize: 500,
    chunkOverlap: 150,
  },
};
