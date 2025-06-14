const { PrivateKey } = require("@hashgraph/sdk");

// Paste your private key from .env here:
const myPrivateKey = PrivateKey.fromStringDer("302e020100300506032b6570042204204e78c1f361f9ff052c366565dad31bceb490977092fb0c093c663701dc573ff7");
console.log("Your Public Key is:", myPrivateKey.publicKey.toString());
