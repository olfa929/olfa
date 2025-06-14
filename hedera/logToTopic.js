require("dotenv").config();
const { Client, TopicMessageSubmitTransaction } = require("@hashgraph/sdk");

const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

// Replace with your topic ID from Step 1
const topicId = "0.0.6163108";

// Example event log (encrypted or hashed if needed)
const message = JSON.stringify({
  patientId: "patient-01",
  event: "Ischemia detected",
  time: new Date().toISOString()
});

async function main() {
  const tx = await new TopicMessageSubmitTransaction()
    .setTopicId(topicId)
    .setMessage(message)
    .execute(client);

  const receipt = await tx.getReceipt(client);
  console.log("✅ Message submitted to topic:", topicId);
}

main();
