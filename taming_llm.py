import os
import time
from dotenv import load_dotenv
import groq
from groq import Groq

# Part 1: Configuration and Basic Completion
class LLMClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Error: GROQ_API_KEY not found in .env file")
            print("Current working directory:", os.getcwd())
            print(".env file exists:", os.path.exists(".env"))
            if os.path.exists(".env"):
                print(".env contents (first line):", open(".env").readline().strip())
            raise ValueError("Missing GROQ API key. Please ensure .env file exists and contains GROQ_API_KEY")
        try:
            self.client = Groq(api_key=self.api_key)
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            print("API key validated successfully:", response)
        except Exception as e:
            print(f"Error validating API key: {str(e)}")
            raise
        self.model = "llama3-70b-8192"

    def complete(self, prompt, max_tokens=1000, temperature=0.7, retries=3):
        """
        Get completion from Groq API with retry logic for rate limits and errors
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                print(f"API Response for prompt '{prompt[:50]}...':", response)
                return response.choices[0].message.content
            except groq.RateLimitError as e:
                print(f"Rate limit hit. Waiting before retry {attempt + 1}/{retries}")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"Error in completion: {str(e)}")
                if attempt == retries - 1:
                    return None
                time.sleep(1)
        return None

# Part 2: Structured Completions
def create_structured_prompt(text, question):
    """
    Creates a structured prompt that will produce a completion with
    easily recognizable sections.
    """
    prompt = f"""
# Analysis Report
## Input Text
{text}
## Question
{question}
## Analysis
"""
    print("Structured prompt created:", prompt)
    return prompt

def extract_section(completion, section_start, section_end=None):
    """
    Extracts content between section_start and section_end.
    If section_end is None, extracts until the end of the completion.
    """
    start_idx = completion.find(section_start)
    if start_idx == -1:
        print(f"Section '{section_start}' not found in completion")
        return None
    start_idx += len(section_start)
    if section_end is None:
        result = completion[start_idx:].strip()
    else:
        end_idx = completion.find(section_end, start_idx)
        if end_idx == -1:
            result = completion[start_idx:].strip()
        else:
            result = completion[start_idx:end_idx].strip()
    print(f"Extracted section from '{section_start}':", result)
    return result

def stream_until_marker(client, prompt, stop_marker, max_tokens=1000):
    """
    Streams the completion and stops once a marker is detected.
    Returns the accumulated text up to the marker.
    """
    print(f"Starting streaming for prompt '{prompt[:50]}...' with stop marker '{stop_marker}'")
    accumulated_text = ""
    try:
        stream = client.client.chat.completions.create(
            model=client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            accumulated_text += content
            print(f"Streaming chunk received:", content)
            if stop_marker in accumulated_text:
                end_idx = accumulated_text.find(stop_marker)
                result = accumulated_text[:end_idx].strip()
                print(f"Stop marker found. Streaming result:", result)
                return result
        print("Streaming completed without finding stop marker")
        return accumulated_text
    except Exception as e:
        print(f"Streaming error: {str(e)}")
        return None

# Part 3: Classification with Confidence Analysis
def classify_with_confidence(client, text, categories, confidence_threshold=0.8):
    """
    Classifies text into one of the provided categories.
    Returns the classification only if confidence is above threshold.
    """
    prompt = f"""
Classify the following text into exactly one of these categories: {', '.join(categories)}.

Response format:
1. CATEGORY: [one of: {', '.join(categories)}]
2. CONFIDENCE: [high|medium|low]
3. REASONING: [explanation]

Text to classify:
{text}
"""
    print(f"Classification prompt for '{text}':", prompt)
    
    try:
        response = client.client.chat.completions.create(
            model=client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        print(f"API Response for classification of '{text}':", response)
        completion = response.choices[0].message.content
        category = extract_section(completion, "1. CATEGORY: ", "\n")
        reasoning = extract_section(completion, "3. REASONING: ", "\n")
        confidence_level = extract_section(completion, "2. CONFIDENCE: ", "\n")
        
        confidence_score = 0.9 if confidence_level == "high" else 0.6 if confidence_level == "medium" else 0.3
        
        result = {
            "category": category,
            "confidence": confidence_score,
            "reasoning": reasoning
        }
        print(f"Classification result for '{text}':", result)
        
        if confidence_score > confidence_threshold:
            return result
        else:
            return {
                "category": "uncertain",
                "confidence": confidence_score,
                "reasoning": "Confidence below threshold"
            }
    except Exception as e:
        print(f"Classification error for '{text}': {str(e)}")
        return {
            "category": "error",
            "confidence": 0.0,
            "reasoning": f"Error during classification: {str(e)}"
        }

# Part 4: Prompt Strategy Comparison
def compare_prompt_strategies(client, texts, categories):
    """
    Compares different prompt strategies on the same classification tasks.
    """
    print("\nComparison of Prompt Strategies:")
    strategies = {
        "basic": lambda text: f"Classify this text into one of these categories: {', '.join(categories)}\nText: {text}\nClassification:",
        "structured": lambda text: f"""
Classification Task
Categories: {', '.join(categories)}
Text: {text}
Classification:""",
        "few_shot": lambda text: f"""
Here are some examples of text classification:
Example 1:
Text: "The product arrived damaged and customer service was unhelpful."
Classification: negative
Example 2:
Text: "While delivery was slow, the quality exceeded my expectations."
Classification: neutral
Example 3:
Text: "Absolutely love this! Best purchase I've made all year."
Classification: positive
Now classify this text:
Text: "{text}"
Classification:"""
    }
    
    results = {}
    for strategy_name, prompt_func in strategies.items():
        print(f"\nStrategy: {strategy_name}")
        strategy_results = []
        for text in texts:
            start_time = time.time()
            prompt = prompt_func(text)
            print(f"Prompt for '{text}':", prompt)
            try:
                response = client.client.chat.completions.create(
                    model=client.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0
                )
                print(f"API Response for '{text}':", response)
                completion = response.choices[0].message.content
                classification = completion.split("Classification:")[-1].strip() if "Classification:" in completion else "unknown"
                result = classify_with_confidence(client, text, categories)
                classification = result["category"]
                confidence = result["confidence"]
                reasoning = result["reasoning"]
            except Exception as e:
                print(f"Error in strategy {strategy_name} for text '{text}': {str(e)}")
                strategy_results.append({
                    "text": text,
                    "classification": "error",
                    "time": 0,
                    "confidence": 0.0,
                    "reasoning": "API error"
                })
                continue
            end_time = time.time()
            time_taken = end_time - start_time
            
            strategy_results.append({
                "text": text,
                "classification": classification,
                "time": time_taken,
                "confidence": confidence,
                "reasoning": reasoning
            })
            print(f"Text: {text}")
            print(f"Classification: {classification}")
            print(f"Confidence: {confidence}")
            print(f"Reasoning: {reasoning}")
            print(f"Time taken: {time_taken:.2f} seconds")
        results[strategy_name] = strategy_results
    
    # Text-based comparison instead of visualization
    print("\nStrategy Comparison Summary:")
    for strategy, data in results.items():
        print(f"\nStrategy: {strategy}")
        category_counts = {"positive": 0, "negative": 0, "neutral": 0, "error": 0, "uncertain": 0}
        total_time = 0
        for result in data:
            category_counts[result["classification"]] = category_counts.get(result["classification"], 0) + 1
            total_time += result["time"]
        avg_time = total_time / len(data) if data else 0
        print("Classification Distribution:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")
        print(f"Average Response Time: {avg_time:.2f} seconds")
    return results

# Main testing code
if __name__ == "__main__":
    try:
        print("Starting execution of taming_llm.py")
        client = LLMClient()
        
        # Test Part 1: Basic Completion
        print("\nTesting Part 1: Basic Completion")
        test_prompt = "Hello, how are you?"
        result = client.complete(test_prompt)
        print("Basic completion result:", result)
        
        # Test Part 2: Structured Completions
        print("\nTesting Part 2: Structured Completions")
        test_text = "The product was amazing and delivery was fast"
        test_question = "What is the sentiment of this review?"
        
        prompt = create_structured_prompt(test_text, test_question)
        completion = client.complete(prompt)
        print("Full completion:", completion)
        
        analysis = extract_section(completion, "## Analysis\n")
        print("Extracted analysis:", analysis)
        
        streaming_result = stream_until_marker(client, prompt, "END")
        print("Streaming result:", streaming_result)
        
        # Test Part 3: Classification
        print("\nTesting Part 3: Classification")
        categories = ["positive", "negative", "neutral"]
        result = classify_with_confidence(client, test_text, categories)
        print("Classification result:")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
        
        # Test Part 4: Prompt Strategies
        print("\nTesting Part 4: Prompt Strategies")
        test_texts = [
            "The product was amazing and delivery was fast",
            "Worst experience ever with customer service",
            "Quality was decent for the price"
        ]
        results = compare_prompt_strategies(client, test_texts, categories)
        print("\nFinal Strategy Results:")
        for strategy, data in results.items():
            print(f"\nStrategy: {strategy}")
            for result in data:
                print(f"Text: {result['text']}")
                print(f"Classification: {result['classification']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Reasoning: {result['reasoning']}")
                print(f"Time: {result['time']:.2f} seconds")
        print("\nExecution completed successfully")
                
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
