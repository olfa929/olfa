require("dotenv").config();
const crypto = require("crypto");
const fs = require("fs");

const key = process.env.ENCRYPTION_KEY;
const iv = Buffer.from(process.env.IV);

function encrypt(text) {
  const cipher = crypto.createCipheriv("aes-256-cbc", Buffer.from(key), iv);
  let encrypted = cipher.update(text);
  encrypted = Buffer.concat([encrypted, cipher.final()]);
  return encrypted.toString("hex");
}

// Load patient data (replace this with real input or JSON)
const patientData = JSON.stringify({
  heartRate: 85,
  glucose: 102,
  diagnosis: "Ischemia",
  timestamp: Date.now()
});

const encrypted = encrypt(patientData);
fs.writeFileSync("encrypted_data.txt", encrypted);
console.log("Encrypted data written to encrypted_data.txt");
