# Document Intelligence System

A sophisticated PDF document analysis tool that extracts structured heading information using semantic classification and hierarchical organization. This system combines layout analysis with machine learning to identify and organize document titles, headings, and subheadings into a clean JSON structure.

## 🚀 Features

- Intelligent PDF Parsing: Extract text with rich metadata including font properties, positioning, and styling
- Semantic Heading Classification: Uses a lightweight ONNX transformer model to classify text into Title, H1, H2, H3, or Other categories
- Hierarchical Structure Building: Automatically organizes headings into logical hierarchies with page references
- Multi-language Support: Handles multilingual documents with robust text processing
- Quality Filtering: Advanced text validation to filter out boilerplate, gibberish, and non-content text
- Containerized Deployment: Ready-to-run Docker container for easy deployment and scaling
- Batch Processing: Process multiple PDF files efficiently with progress tracking

## 🏃‍♂️ Quick Start

### Using Docker (Recommended)

1. Build the Docker image:
  
bash
   docker build --platform linux/amd64 -t document-intelligence:latest .
   

2. Run with your PDF files:
  
bash
   docker run -v /path/to/your/pdfs:/app/input -v /path/to/output:/app/output document-intelligence:latest
   

3. Check the output:
  
bash
   ls /path/to/output/
   # You'll find JSON files with extracted heading structures
   

## 📦 Installation

### Docker Installation (Recommended)

bash
# Clone the repository
git clone https://github.com/aryamagarwal/adobe-1a-stacksmith.git
cd adobe-1a-stacksmith

# Build the Docker image
docker build --platform linux/amd64 -t document-intelligence:latest .

# Run the container
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output document-intelligence:latest


### Local Installation

bash
# Clone the repository
git clone https://github.com/aryamagarwal/adobe-1a-stacksmith.git
cd adobe-1a-stacksmith

# Install Python dependencies
pip install -r requirements.txt

# Run the application
cd app
python main.py


## 💡 Usage

### Directory Structure


your-project/
├── input/          # Place your PDF files here
├── output/         # Processed JSON files will appear here
└── app/
    ├── main.py     # Main application
    └── onnx_model/ # Pre-trained ONNX model and tokenizer


### Input Format

Place PDF files in the input/ directory. The system supports:
- Standard PDF documents
- Multi-page documents
- Documents with complex layouts
- Multilingual content

### Output Format

The system generates JSON files in the output/ directory with the following structure:

json
{
  "filename": "document.pdf",
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1: Introduction",
      "page": 1,
      "confidence": 0.95
    },
    {
      "level": "H2",
      "text": "1.1 Background",
      "page": 2,
      "confidence": 0.87
    },
    {
      "level": "H3",
      "text": "1.1.1 Methodology",
      "page": 2,
      "confidence": 0.82
    }
  ],
  "processing_stats": {
    "total_pages": 10,
    "headings_found": 15,
    "processing_time": "2.3s"
  }
}


### Example Usage

bash
# Process a single document
docker run -v /home/user/documents:/app/input -v /home/user/results:/app/output document-intelligence:latest

# Process multiple documents
# Place multiple PDFs in the input directory and run the same command


## 🏗️ Technical Architecture

### Processing Pipeline

1. PDF Text Extraction: Uses PyMuPDF to extract text spans with comprehensive metadata
2. Chunk Formation: Segments pages into meaningful text chunks based on font properties and layout
3. Quality Filtering: Applies multiple filters to remove boilerplate, navigation, and low-quality text
4. Semantic Classification: Uses a distilled ONNX transformer model for heading-level classification
5. Confidence Scoring: Calculates confidence scores based on semantic similarity and layout features
6. Hierarchy Building: Constructs logical heading hierarchies with proper nesting

### ONNX Model Details

- Model: Distilled transformer model (~40MB)
- Architecture: Based on sentence-transformers architecture
- Training: Optimized for multilingual heading classification
- Inference: Fast ONNX runtime for production deployment
- Quantization: INT8 quantized for optimal performance

### Text Processing Features

- Font Analysis: Leverages font size, weight, and style for heading detection
- Position Awareness: Uses spatial positioning for layout understanding
- Context Enhancement: Refines vague headings using document context
- Boilerplate Detection: Advanced pattern matching to filter non-content text

## 📋 API Reference

### Input Parameters

The system processes all PDF files found in the /app/input directory within the container.

### Output Schema

json
{
  "filename": "string",           // Original PDF filename
  "title": "string|null",         // Extracted document title
  "outline": [                    // Array of heading objects
    {
      "level": "Title|H1|H2|H3",  // Heading level classification
      "text": "string",           // Cleaned heading text
      "page": "integer",          // Page number (1-indexed)
      "confidence": "float"       // Classification confidence (0-1)
    }
  ],
  "processing_stats": {
    "total_pages": "integer",     // Total pages processed
    "headings_found": "integer",  // Total headings extracted
    "processing_time": "string"   // Processing duration
  }
}


## ⚙️ Configuration

### Environment Variables

- INPUT_DIR: Input directory path (default: /app/input)
- OUTPUT_DIR: Output directory path (default: /app/output)
- LOG_LEVEL: Logging level (default: INFO)

### Model Configuration

The ONNX model and tokenizer are pre-configured and included in the Docker image:
- Model: app/onnx_model/model_quantized.onnx
- Tokenizer: app/onnx_model/tokenizer.json
- Config: app/onnx_model/config.json

### Customization Parameters

Key parameters can be adjusted in main.py:
- SEMANTIC_THRESHOLD: Minimum confidence for heading classification (default: 1.5)
- MIN_HEADING_LENGTH: Minimum character length for headings (default: 1)
- MAX_HEADING_LENGTH: Maximum word count for headings (default: 10)

## 📚 Dependencies

### Core Libraries

- PyMuPDF (1.22.5): Advanced PDF parsing and text extraction
- onnxruntime (1.17.1): High-performance ONNX model inference
- tokenizers (0.13.3): Fast tokenization for transformer models
- numpy (1.24.2): Numerical operations and array processing
- tqdm (4.65.0): Progress tracking for batch processing
- huggingface-hub (0.14.1): Model and tokenizer management

### System Requirements

- Python: 3.10 or higher
- Memory: Minimum 2GB RAM (4GB recommended for large documents)
- Storage: 500MB for Docker image
- Platform: Linux/AMD64 (Docker), Windows/macOS/Linux (local)

## 🤝 Contributing

We welcome contributions to improve the Document Intelligence System! Here's how you can help:

### Development Setup

bash
# Fork and clone the repository
git clone https://github.com/yourusername/adobe-1a-stacksmith.git
cd adobe-1a-stacksmith

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/


### Contributing Guidelines
1. Fork the repository
2. Create a feature branch: git checkout -b feature-name
3. Make your changes with clear, commented code
4. Test your changes thoroughly
5. Commit with descriptive messages: git commit -m "Add feature: description"
6. Push to your fork: git push origin feature-name
7. Submit a pull request with detailed description

### Areas for Contribution

- Model Improvements: Enhanced heading classification models
- Language Support: Additional language-specific processing
- Output Formats: Support for XML, CSV, or other formats
- Performance Optimization: Speed and memory usage improvements
- Documentation: Examples, tutorials, and API documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


MIT License

Copyright (c) 2025 Aryam Agarwal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.


## 🙏 Acknowledgments

- PyMuPDF Team: For the excellent PDF parsing library
- ONNX Runtime Team: For high-performance model inference
- Hugging Face: For transformer models and tokenization tools
- sentence-transformers: For the base model architecture
- Community Contributors: For feedback, testing, and improvements

---

Built by Team Stacksmiths

For questions, issues, or feature requests, please visit our [GitHub repository](https://github.com/aryamagarwal/adobe-1a-stacksmith) or open an issue.
