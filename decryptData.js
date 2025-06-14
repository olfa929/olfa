require("dotenv").config();
const crypto = require("crypto");
const fs = require("fs");

const key = process.env.ENCRYPTION_KEY;
const iv = Buffer.from(process.env.IV);

function decrypt(text) {
  const encryptedText = Buffer.from(text, "hex");
  const decipher = crypto.createDecipheriv("aes-256-cbc", Buffer.from(key), iv);
  let decrypted = decipher.update(encryptedText);
  decrypted = Buffer.concat([decrypted, decipher.final()]);
  return decrypted.toString();
}

const encrypted = fs.readFileSync("encrypted_data.txt", "utf8");
const decrypted = decrypt(encrypted);
console.log("Decrypted content:", decrypted);
