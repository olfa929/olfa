require("dotenv").config();
const { Client, FileUpdateTransaction, Hbar } = require("@hashgraph/sdk");

// Setup client
const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

// Your file ID from earlier
const fileId = "0.0.6163016";

async function main() {
  const transaction = await new FileUpdateTransaction()
    .setFileId(fileId)
    .setKeys([]) // Removing all keys = public
    .setMaxTransactionFee(new Hbar(2))
    .freezeWith(client);

  const signTx = await transaction.signWithOperator(client);
  const txResponse = await signTx.execute(client);
  const receipt = await txResponse.getReceipt(client);

  console.log(`File ${fileId} updated to be publicly viewable.`);
}

main();
