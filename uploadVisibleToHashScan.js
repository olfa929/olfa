require("dotenv").config();
const { Client, FileCreateTransaction, Hbar, PrivateKey, PublicKey } = require("@hashgraph/sdk");
const fs = require("fs");

const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

// Replace this with your real public key
const myPublicKey = PublicKey.fromString("302a300506032b65700321009f016e6f93a1290f37830140a03c75361e2c44fe5e9db29d9eb25e26ebe4b34d");

const encryptedData = fs.readFileSync("encrypted_data.txt", "utf8");

async function main() {
  const transaction = await new FileCreateTransaction()
    .setKeys([myPublicKey]) // 🔓 Now you can update it later
    .setContents(encryptedData)
    .setMaxTransactionFee(new Hbar(2))
    .freezeWith(client);

  const signedTx = await transaction.signWithOperator(client);
  const txResponse = await signedTx.execute(client);
  const receipt = await txResponse.getReceipt(client);

  console.log("🆕 File ID visible to HashScan:", receipt.fileId.toString());
}

main();
