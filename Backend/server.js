//main point that connected to the server(Mongo)
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import connectDB from "./config/mongooseDb.js";

dotenv.config();        // Load .env file
connectDB();            // Connect to MongoDB

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Test route
app.get("/", (req, res) => {
  res.send("Sign Language Translator Backend is running + MongoDB connected!");
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});