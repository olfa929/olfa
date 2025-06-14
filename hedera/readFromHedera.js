require("dotenv").config();
const { Client, FileContentsQuery } = require("@hashgraph/sdk");

const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

const fileId = "0.0.6163016";

async function main() {
  const fileContents = await new FileContentsQuery()
    .setFileId(fileId)
    .execute(client);

  console.log("Encrypted data retrieved from Hedera:");
  console.log(fileContents.toString());
}

main();
