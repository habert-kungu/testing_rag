import { ingestDocument } from "./loadFile";
import { queryData } from "./ragTool";

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log("Please provide a command: 'ingest' or 'query'");
    return;
  }

  const command = args[0];

  let vectorStore: any; // Declare vectorStore here

  if (command === "ingest") {
    if (args.length < 2) {
      console.log("Please provide the path to the document to ingest.");
      return;
    }
    const filePath = args[1];
    vectorStore = await ingestDocument(filePath);
  } else if (command === "query") {
    if (args.length < 2) {
      console.log("Please provide a query.");
      return;
    }
    const query = args.slice(1).join(" ");
    // Automatically ingest the default data.json for in-memory usage
    console.log("Ingesting data.json for query...");
    vectorStore = await ingestDocument("src/sample_files/data.json");
    const result = await queryData(query, vectorStore);
    console.log("Answer:");
    console.log(result);
    process.exit(0);
  } else {
    console.log(`Unknown command: ${command}`);
  }
}

main().catch(console.error);
