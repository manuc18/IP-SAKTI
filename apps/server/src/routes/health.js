import { Router } from "express";
import { getHealth } from "../services/pipelineService.js";

const router = Router();

router.get("/health", async (req, res) => {
  const { statusCode, data } = await getHealth();
  res.status(statusCode).json(data);
});

router.get("/api/health", async (req, res) => {
  const { statusCode, data } = await getHealth();
  res.status(statusCode).json(data);
});

export default router;
