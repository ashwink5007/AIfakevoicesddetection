const express = require('express');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const router = express.Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const uploadsDir = path.join(__dirname, '../uploads');

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedMimes = [
      'audio/wav',
      'audio/mpeg',
      'audio/flac',
      'audio/ogg',
      'audio/mp4',
      'audio/webm'
    ];
    
    if (allowedMimes.includes(file.mimetype) || file.originalname.match(/\.(wav|mp3|flac|ogg|m4a|webm)$/i)) {
      cb(null, true);
    } else {
      cb(new Error('Only audio files are allowed'));
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// POST /api/audio/predict - Upload and predict
router.post('/predict', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filePath = req.file.path;
    const fileName = req.file.originalname;

    // Read file and send to ML service
    const fileStream = fs.createReadStream(filePath);

    const formData = new FormData();
    const blob = await new Promise((resolve, reject) => {
      const chunks = [];
      fileStream.on('data', chunk => chunks.push(chunk));
      fileStream.on('end', () => resolve(new Blob(chunks, { type: req.file.mimetype })));
      fileStream.on('error', reject);
    });

    formData.append('file', blob, fileName);

    // Send to FastAPI ML service
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000 // 60 seconds timeout
    });

    // Clean up uploaded file after processing
    fs.unlink(filePath, (err) => {
      if (err) console.error('Error deleting file:', err);
    });

    // Return prediction with metadata
    res.json({
      success: true,
      prediction: mlResponse.data.prediction,
      confidence: mlResponse.data.confidence,
      fileName: fileName,
      uploadedAt: new Date().toISOString()
    });

  } catch (error) {
    // Clean up file on error
    if (req.file) {
      fs.unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting file:', err);
      });
    }

    console.error('Prediction error:', error.message);

    if (error.response?.status === 503) {
      return res.status(503).json({
        error: 'ML Model not found. Train model first.'
      });
    }

    res.status(error.response?.status || 500).json({
      error: error.response?.data?.detail || error.message || 'Prediction failed'
    });
  }
});

// GET /api/audio/health - Health check for audio service
router.get('/health', async (req, res) => {
  try {
    const mlHealth = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 5000 });
    res.json({
      backend: 'ok',
      ml_service: 'ok',
      ml_model_loaded: mlHealth.data.model_loaded
    });
  } catch (error) {
    res.status(503).json({
      backend: 'ok',
      ml_service: 'unavailable',
      error: error.message
    });
  }
});

module.exports = router;
