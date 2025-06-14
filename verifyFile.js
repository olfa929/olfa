require("dotenv").config();
const { Client, FileInfoQuery } = require("@hashgraph/sdk");

const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

const fileId = "0.0.6163092";

async function main() {
  const info = await new FileInfoQuery()
    .setFileId(fileId)
    .execute(client);

  console.log("✅ File info retrieved:");
  console.log("File ID:", fileId);
  console.log("Size:", info.size, "bytes");
  console.log("Deleted:", info.isDeleted);
  console.log("Keys:", info.keys.toString());
}

main();
