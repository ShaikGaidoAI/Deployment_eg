# Hugging Face Model Setup

This project demonstrates how to use Hugging Face models through LangChain.

## Setup Instructions

1. Install the required packages:
   ```
   pip install langchain langchain_huggingface
   ```

2. Get a Hugging Face API token:
   - Create an account at [Hugging Face](https://huggingface.co/) if you don't have one
   - Go to your profile > Settings > Access Tokens
   - Create a new token with read access

3. Set up your token:
   - Run `python setup_token.py` and enter your token when prompted
   - Or set it as an environment variable: `HUGGINGFACE_API_TOKEN=your_token`

## Usage

Run the model with:
```
python model_setup.py
```

## Notes

- For Gemma models, make sure to use `task='text-generation'`
- The supported tasks for HuggingFaceHub are: 'translation', 'summarization', 'conversational', 'text-generation', and 'text2text-generation'
