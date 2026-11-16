import { Router } from "express";
import { askQuestion } from "../services/pipelineService.js";

const router = Router();

router.post("/ask", async (req, res) => {
  try {
    const { question, clarified } = req.body || {};
    const { statusCode, data } = await askQuestion(question, { clarified: clarified === true });
    return res.status(statusCode).json(data);
  } catch (err) {
    return res.status(500).json({
      success: false,
      status: "error",
      message: "The IP-SAKTI service could not process the request.",
      error_code: "INTERNAL_SERVER_ERROR",
    });
  }
});

export default router;
