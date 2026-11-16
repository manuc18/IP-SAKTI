import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 1. Load apps/server/.env using explicit directory path
dotenv.config({ path: path.resolve(__dirname, "../.env") });

// 2. Load root workspace .env
dotenv.config({ path: path.resolve(__dirname, "../../../.env") });

// 3. Process current working directory .env
dotenv.config();
