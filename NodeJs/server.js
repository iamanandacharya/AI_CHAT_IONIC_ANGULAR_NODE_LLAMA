const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
const PORT = 8000;

app.use(cors());
app.use(express.json());

// Route to handle chat requests
app.post("/generate", async (req, res) => {
    const { prompt } = req.body;
    console.log(prompt)
    try {
        // Call Python backend
        const response = await axios.post("http://127.0.0.1:8001/generate", { prompt });
        res.json({ response: response.data.response });
    } catch (error) {
        console.error("Error calling AI backend:", error.message);
        res.status(500).json({ error: "Failed to get response from AI backend" });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
