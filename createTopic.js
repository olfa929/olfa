require("dotenv").config();
const { Client, TopicCreateTransaction, Hbar } = require("@hashgraph/sdk");

const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

async function main() {
  const txResponse = await new TopicCreateTransaction()
    .setAdminKey(client.operatorPublicKey) // So only you can manage the topic
    .setSubmitKey(client.operatorPublicKey) // So only you can submit messages
    .setTopicMemo("Heart Digital Twin Logs")
    .setMaxTransactionFee(new Hbar(2))
    .execute(client);

  const receipt = await txResponse.getReceipt(client);
  const topicId = receipt.topicId;

  console.log("✅ Topic created:", topicId.toString());
}

main();
