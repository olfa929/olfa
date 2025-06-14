require("dotenv").config();
const { Client, FileCreateTransaction, Hbar } = require("@hashgraph/sdk");
const fs = require("fs");

// Setup client
const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

// Load encrypted data
const encryptedData = fs.readFileSync("encrypted_data.txt", "utf8");

async function main() {
  const transaction = new FileCreateTransaction()
    .setKeys([])  // Leave empty for public file, or set access keys for control
    .setContents(encryptedData)
    .setMaxTransactionFee(new Hbar(2))
    .freezeWith(client);

  const txResponse = await transaction.execute(client);
  const receipt = await txResponse.getReceipt(client);
  const fileId = receipt.fileId;

  console.log(`Encrypted data stored on Hedera with File ID: ${fileId}`);
}

main();
