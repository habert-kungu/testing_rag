import { ingestDocument } from "./loadFile";
import { queryData } from "./ragTool";

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log("Please provide a command: 'ingest' or 'query'");
    return;
  }

  const command = args[0];

  if (command === "ingest") {
    if (args.length < 2) {
      console.log("Please provide the path to the document to ingest.");
      return;
    }
    const filePath = args[1];
    await ingestDocument(filePath);
  } else if (command === "query") {
    if (args.length < 2) {
      console.log("Please provide a query.");
      return;
    }
    const query = args.slice(1).join(" ");
    const result = await queryData(query);
    console.log("Answer:");
    console.log(result);
  } else {
    console.log(`Unknown command: ${command}`);
  }
}

main().catch(console.error);