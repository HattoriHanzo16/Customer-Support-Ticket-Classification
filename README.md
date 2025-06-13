# Customer Support Ticket Classification System 🎯

DEMO [https://www.youtube.com/shorts/mRJnD8weHaY]

A machine learning system that automatically categorizes customer support tickets into appropriate departments using OpenAI's fine-tuning capabilities. This project demonstrates how to effectively use fine-tuning to improve classification accuracy in a real-world business scenario.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Categories](#categories)
- [Results and Analysis](#results-and-analysis)
- [Technical Details](#technical-details)
- [Best Practices](#best-practices)
- [Lessons Learned](#lessons-learned)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements an automated customer support ticket classification system that can accurately categorize incoming support requests into five main categories: Technical, Billing, Account, Product, or Other. The system uses OpenAI's fine-tuning capabilities to improve upon the baseline GPT-3.5-turbo model's performance.

### Key Achievements
- Achieved 100% accuracy on test data after fine-tuning
- 26.67% improvement over baseline model
- Successfully resolved category confusion issues
- Implemented with a relatively small but effective dataset

## ✨ Features

- **Automated Classification**: Quickly and accurately categorizes support tickets
- **Multiple Categories**: Handles five distinct support categories
- **High Accuracy**: Achieves perfect accuracy on test data
- **Easy Integration**: Simple API-based implementation
- **Cost-Effective**: Uses fine-tuned smaller model for better efficiency
- **Detailed Analytics**: Provides comprehensive performance metrics

## 📁 Project Structure

```
.
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/                     # Data directory
│   ├── training_data.jsonl   # Training dataset
│   └── test_data.jsonl       # Test dataset
├── src/                      # Source code
│   ├── generate_data.py      # Data generation script
│   ├── fine_tune.py         # Fine-tuning implementation
│   └── evaluate.py          # Model evaluation script
└── .env                      # Environment variables
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-support-classifier.git
   cd customer-support-classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

## 💻 Usage

### 1. Generate Training Data
```bash
python src/generate_data.py
```
This creates:
- 50 training examples (10 per category)
- 15 test examples (3 per category)

### 2. Run Baseline Evaluation
```bash
python src/evaluate.py --baseline
```
Evaluates the base GPT-3.5-turbo model's performance.

### 3. Fine-tune the Model
```bash
python src/fine_tune.py
```
Creates and monitors the fine-tuning job.

### 4. Evaluate Fine-tuned Model
```bash
python src/evaluate.py
```
Compares fine-tuned model performance with baseline.

## 🏷️ Categories

The system classifies tickets into these categories:

| Category | Description | Example Scenarios |
|----------|-------------|-------------------|
| Technical | Hardware/software issues | System errors, crashes, connectivity problems |
| Billing | Payment-related issues | Refunds, charges, subscription management |
| Account | User account issues | Login problems, password resets, access issues |
| Product | Product-related queries | Features, setup, compatibility questions |
| Other | General inquiries | Feedback, company information, partnerships |

## 📊 Results and Analysis

### Baseline Model Performance
- **Overall Accuracy**: 73.33%
- **Category Performance**:
  - Account: 33.33% recall (1.00 precision)
  - Billing: 100% recall and precision
  - Other: 66.67% recall (1.00 precision)
  - Product: 66.67% recall (1.00 precision)
  - Technical: 100% recall (0.43 precision)

### Fine-tuned Model Performance
- **Overall Accuracy**: 100%
- **Perfect Performance** across all categories
- **Key Improvements**:
  - Resolved Technical vs. Account confusion
  - Better Product category distinction
  - Improved Other category recognition

## 🔧 Technical Details

### Model Architecture
- Base Model: GPT-3.5-turbo
- Fine-tuned Version: Custom model for ticket classification
- Training Data: 50 examples (JSONL format)
- Test Data: 15 examples

### Data Format
```json
{
    "messages": [
        {"role": "system", "content": "You are a customer support ticket classifier..."},
        {"role": "user", "content": "Customer message here"},
        {"role": "assistant", "content": "Category"}
    ]
}
```

## 🎯 Best Practices

### When to Use Fine-tuning
- Large, high-quality dataset available
- Domain-specific knowledge required
- Need for consistent categorization
- Cost optimization needed
- Baseline model shows category confusion

### When to Use Better Prompts
- Limited training data
- Simple classification tasks
- Need for flexible modifications
- Edge case handling required
- Baseline performance is already good

## 📚 Lessons Learned

1. **Data Quality Matters**
   - Small but well-structured datasets can be effective
   - Diverse examples within categories are crucial
   - Real-world scenarios improve model performance

2. **Category Design**
   - Clear category boundaries are essential
   - Some categories naturally overlap (Technical vs. Product)
   - System message design impacts classification

3. **Fine-tuning Benefits**
   - Significant accuracy improvements possible
   - Resolves category confusion effectively
   - Cost-effective for production use

4. **Implementation Insights**
   - Proper error handling is crucial
   - Monitoring fine-tuning progress is important
   - Regular evaluation helps track improvements

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the API and fine-tuning capabilities
- The open-source community for various tools and libraries
- Contributors and users of this project

