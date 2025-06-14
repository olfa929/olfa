require("dotenv").config();
const { Client, TopicMessageQuery } = require("@hashgraph/sdk");

const client = Client.forTestnet();
client.setOperator(process.env.MY_ACCOUNT_ID, process.env.MY_PRIVATE_KEY);

const topicId = "0.0.6163108";

async function main() {
  new TopicMessageQuery()
    .setTopicId(topicId)
    .subscribe(client, null, (msg) => {
      console.log("📥 Received log:", Buffer.from(msg.contents, "utf8").toString());
    });
}

main();
