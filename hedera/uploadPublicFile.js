require("dotenv").config();
const { Client, FileCreateTransaction, Hbar } = require("@hashgraph/sdk");
const fs = require("fs");

// Setup
const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

// Load encrypted data
const encryptedData = fs.readFileSync("encrypted_data.txt", "utf8");

async function main() {
  const transaction = await new FileCreateTransaction()
    .setKeys([]) // No keys = fully public
    .setContents(encryptedData)
    .setMaxTransactionFee(new Hbar(2))
    .freezeWith(client);

  const signedTx = await transaction.signWithOperator(client);
  const txResponse = await signedTx.execute(client);
  const receipt = await txResponse.getReceipt(client);

  const fileId = receipt.fileId;
  console.log(`✅ Public file uploaded. File ID: ${fileId}`);
}

main();
