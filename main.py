import os
from dotenv import load_dotenv
from backend.app.engines.hallucination_hunter import HallucinationHunter

load_dotenv()

def main():
    print("Starting LLM Fragility Lab...")
    hunter = HallucinationHunter()

    response = "The capital of France is Paris."
    ground_truth = "Paris is the capital of France."

    print(f"Analyzing response: '{response}' against ground truth: '{ground_truth}'")
    result = hunter.analyze(response, ground_truth)

    print(f"Result: {result}")

if __name__ == "__main__":
    main()
