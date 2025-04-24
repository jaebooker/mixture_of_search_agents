import math
import random
import requests
import json
import time
import os
from typing import List, Dict, Optional
import numpy as np
from mcts_llm import MCTS, Node, LLMClient, generate_actions, finalize_sub_answer

class LLMClient:
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, local: bool = False):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or self._get_default_base_url()
        self.local = local
        
    def _get_default_base_url(self) -> str:
        urls = {
            "llama-3.1-8b-instruct": "https://api.replicate.com/v1/predictions",
            "qwen-2-7b-instruct": "https://api.together.xyz/v1/chat/completions",
            "mistral-8b-instruct": "https://api.mistral.ai/v1/chat/completions",
            "glm-4-9b-chat": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            "local-llama2": "http://localhost:11434/api/generate",
            "local-mistral": "http://localhost:11434/api/generate"
        }
        return urls.get(self.model_name.lower())

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.local:
            return self._generate_local(prompt, max_tokens)
        else:
            return self._generate_api(prompt, max_tokens)
            
    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            if self.model_name.lower() == "local-llama2":
                data = {
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.7
                    }
                }
            elif self.model_name.lower() == "local-mistral":
                data = {
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.7
                    }
                }
            else:
                raise ValueError(f"Unsupported local model: {self.model_name}")
                
            response = requests.post(self.base_url, headers=headers, json=data)
            response_data = response.json()
            
            if "response" in response_data:
                return response_data["response"]
            else:
                raise Exception(f"Unexpected response format from Ollama: {response_data}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Local API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse local API response: {str(e)}")
            
    def _generate_api(self, prompt: str, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            if self.model_name.lower() == "llama-3.1-8b-instruct":
                data = {
                    "version": "meta/llama-2-70b-chat",
                    "input": {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.7
                    }
                }
                create_response = requests.post(self.base_url, headers=headers, json=data)
                create_data = create_response.json()
                print("Replicate API Create Response:", create_data)
                
                if "error" in create_data and create_data["error"] is not None:
                    raise Exception(f"Replicate API error: {create_data['error']}")
                
                prediction_id = create_data.get("id")
                if not prediction_id:
                    raise Exception(f"No prediction ID in Replicate API response: {create_data}")
                
                polling_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
                
                max_attempts = 30  
                attempt = 0
                
                while attempt < max_attempts:
                    response = requests.get(polling_url, headers=headers)
                    response_data = response.json()
                    print("Replicate API Poll Response:", response_data)
                    
                    if response_data["status"] == "succeeded":
                        if "output" in response_data and response_data["output"] is not None:
                            return response_data["output"]
                        else:
                            raise Exception("No output in successful Replicate API response")
                    elif response_data["status"] == "failed":
                        error_msg = response_data.get("error", "Unknown error")
                        raise Exception(f"Replicate API prediction failed: {error_msg}")
                    elif response_data["status"] in ["starting", "processing"]:
                        time.sleep(2)
                        attempt += 1
                    else:
                        raise Exception(f"Unexpected status from Replicate API: {response_data['status']}")
                
                raise Exception("Replicate API polling timeout - maximum attempts reached")
                
            elif self.model_name.lower() == "qwen-2-7b-instruct":
                data = {
                    "model": "mistralai/Mistral-7B-Instruct-v0.1",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.7
                }
                response = requests.post(self.base_url, headers=headers, json=data)
                response_data = response.json()
                print("Together API Response:", response_data)
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    return response_data["choices"][0]["message"]["content"]
                elif "error" in response_data:
                    raise Exception(f"Together API error: {response_data['error']}")
                else:
                    raise Exception("Unexpected response format from Together API")
                
            elif self.model_name.lower() == "mistral-8b-instruct":
                data = {
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                response = requests.post(self.base_url, headers=headers, json=data)
                response_data = response.json()
                print("Mistral API Response:", response_data)  
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    if "message" in response_data["choices"][0]:
                        return response_data["choices"][0]["message"]["content"]
                    elif "text" in response_data["choices"][0]:
                        return response_data["choices"][0]["text"]
                elif "error" in response_data:
                    raise Exception(f"Mistral API error: {response_data['error']}")
                else:
                    raise Exception(f"Unexpected response format from Mistral API: {response_data}")
                
            elif self.model_name.lower() == "glm-4-9b-chat":
                data = {
                    "model": "glm-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.7
                }
                response = requests.post(self.base_url, headers=headers, json=data)
                response_data = response.json()
                print("Zhipu API Response:", response_data)  # Debug print
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    return response_data["choices"][0]["message"]["content"]
                elif "error" in response_data:
                    raise Exception(f"Zhipu API error: {response_data['error']}")
                else:
                    raise Exception("Unexpected response format from Zhipu API")
                
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {str(e)}")

def select_llm(llm_mix: List[LLMClient]) -> LLMClient:
    return random.choice(llm_mix)

def generate_sub_question(llm: LLMClient, state: str) -> str:
    prompt = f"""Given the current reasoning state: '{state}'
Generate a relevant sub-question that would help progress the reasoning process.
Focus on breaking down the problem into smaller, more manageable parts.
Return only the sub-question without any additional explanation."""
    
    return llm.generate(prompt)

def generate_sub_answer(llm: LLMClient, state: str, sub_question: str) -> str:
    prompt = f"""Given the current reasoning state: '{state}'
And the sub-question: '{sub_question}'
Provide a clear and concise answer that helps progress the reasoning process.
Return only the answer without any additional explanation."""

    return llm.generate(prompt)

aggregator_llm = LLMClient(
    model_name="local-mistral",
    local=True
)

def finalize_sub_answer(candidate_sub_answers, use_aggregator=False):
    str_answers = []
    for answer in candidate_sub_answers:
        if isinstance(answer, list):
            str_answers.append(" ".join(str(item) for item in answer))
        else:
            str_answers.append(str(answer))

    if use_aggregator:
        prompt = f"""You are an expert answer aggregator. Your task is to analyze multiple candidate answers and synthesize them into a single, coherent, and accurate answer.

Candidate answers:
{chr(10).join(f"- {answer}" for answer in str_answers)}

Please analyze these answers and provide a single, well-reasoned answer that:
1. Combines the best elements from each candidate
2. Resolves any contradictions
3. Provides the most accurate and complete information
4. Is concise and clear

Your synthesized answer:"""

        try:
            response = aggregator_llm.generate(prompt)
            if response and isinstance(response, str):
                return f"Aggregated: {response}"
            else:
                print("Warning: Aggregator returned invalid response, falling back to simple aggregation")
                return "Aggregated(" + " | ".join(str_answers) + ")"
        except Exception as e:
            print(f"Error in aggregator: {e}")
            return "Aggregated(" + " | ".join(str_answers) + ")"
    else:
        return str_answers[0] if str_answers else "No Answer"

def generate_actions(state, num_sub_questions, num_candidate_sub_answers, llm_mix, use_aggregator=False):
    new_actions = []
    for i in range(num_sub_questions):
        llm_sub_q = select_llm(llm_mix)
        sub_question = generate_sub_question(llm_sub_q, state)
        
        candidate_sub_answers = []
        for j in range(num_candidate_sub_answers):
            llm_sub_a = select_llm(llm_mix)
            candidate = generate_sub_answer(llm_sub_a, state, sub_question)
            candidate_sub_answers.append(candidate)
        
        sub_answer = finalize_sub_answer(candidate_sub_answers, use_aggregator)
        
        action = f"{sub_question}\n{sub_answer}"
        new_actions.append(action)
        
    return new_actions

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.q_value = 0.0
        self.is_terminal = False
        self.depth = parent.depth + 1 if parent else 0

class MCTS:
    def __init__(self, root, llm_mix, num_sub_questions, num_candidate_sub_answers,
                 use_aggregator=False, max_depth=3, iterations=100):
        self.root = root
        self.llm_mix = llm_mix
        self.num_sub_questions = num_sub_questions
        self.num_candidate_sub_answers = num_candidate_sub_answers
        self.use_aggregator = use_aggregator
        self.max_depth = max_depth
        self.iterations = iterations
        self.answer_history = {}

    def uct_score(self, child, parent_visits, c=1.4):
        if child.visits == 0:
            return float('inf')
        return (child.q_value / child.visits) + c * math.sqrt(math.log(parent_visits) / child.visits)

    def select(self, node):
        path = []
        while node.children:
            path.append(node)
            node = max(node.children, key=lambda child: self.uct_score(child, node.visits))
        path.append(node)
        return path

    def expand(self, node):
        if node.depth < self.max_depth:
            actions = generate_actions(node.state, self.num_sub_questions,
                                        self.num_candidate_sub_answers, self.llm_mix, use_aggregator=self.use_aggregator)
            for action in actions:
                new_state = node.state + "\n" + action
                child = Node(new_state, parent=node, action=action)
                if child.depth == self.max_depth:
                    child.is_terminal = True
                node.children.append(child)

    def _calculate_majority_confidence(self, answers):
        if not answers:
            return 0.0
            
        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
        if not answer_counts:
            return 0.0
            
        max_count = max(answer_counts.values())
        total_count = len(answers)
        
        confidence = max_count / total_count
        
        for answer, count in answer_counts.items():
            self.answer_history[answer] = self.answer_history.get(answer, 0) + count
            
        return confidence

    def simulate(self, node):
        answers = []
        current_node = node
        while current_node:
            if current_node.action:
                parts = current_node.action.split('\n')
                if len(parts) > 1:
                    answers.append(parts[1])
            current_node = current_node.parent
            
        confidence = self._calculate_majority_confidence(answers)
        
        reward = confidence
        
        return reward

    def backpropagate(self, path, reward):
        for node in reversed(path):
            node.visits += 1
            node.q_value += reward

    def run(self):
        for i in range(self.iterations):
            path = self.select(self.root)
            leaf = path[-1]
            if not leaf.is_terminal:
                self.expand(leaf)
                if leaf.children:
                    leaf = random.choice(leaf.children)
                    path.append(leaf)
            reward = self.simulate(leaf)
            self.backpropagate(path, reward)
        return self.root

    def best_action(self):
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda n: n.visits)
        return best_child.action, best_child.state

class MATH500Dataset:
    def __init__(self, data_path: str = None, split: str = "train", max_samples: int = 100):
        self.data_path = data_path
        self.split = split
        self.max_samples = max_samples
        self.samples = self._load_samples()
        
    def _create_sample_dataset(self) -> List[Dict]:
        return [
            {
                "problem": "What is 2 + 2?",
                "solution": "2 + 2 = 4",
                "answer": "4",
                "level": "basic",
                "type": "arithmetic"
            },
            {
                "problem": "If a rectangle has length 5 and width 3, what is its area?",
                "solution": "Area = length * width = 5 * 3 = 15",
                "answer": "15",
                "level": "basic",
                "type": "geometry"
            },
            {
                "problem": "Solve for x: 2x + 5 = 15",
                "solution": "2x + 5 = 15\n2x = 15 - 5\n2x = 10\nx = 5",
                "answer": "5",
                "level": "basic",
                "type": "algebra"
            }
        ]
        
    def _load_samples(self) -> List[Dict]:
        samples = []
        try:
            if self.data_path and os.path.exists(self.data_path):
                with open(self.data_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if "split" not in data or data.get("split") == self.split:
                                processed = {
                                    "problem": data.get("problem", ""),
                                    "solution": data.get("solution", ""),
                                    "answer": data.get("answer", ""),
                                    "level": data.get("level", ""),
                                    "type": data.get("type", "")
                                }
                                samples.append(processed)
                                
                                if self.max_samples and len(samples) >= self.max_samples:
                                    break
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {e}")
                            continue
                            
                if not samples:
                    print("No valid samples found in JSONL file. Using sample dataset.")
                    samples = self._create_sample_dataset()
            else:
                print("No dataset file found. Using sample dataset.")
                samples = self._create_sample_dataset()
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using sample dataset instead.")
            samples = self._create_sample_dataset()
            
        if not samples:
            print("Warning: No samples available. Using sample dataset.")
            samples = self._create_sample_dataset()
            
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

def evaluate_answer(predicted: str, ground_truth: str) -> float:
    return 1.0 if predicted.strip() == ground_truth.strip() else 0.0

def evaluate_answer_with_llm(predicted: str, ground_truth: str, llm: LLMClient) -> float:
    prompt = f"""You are a math expert evaluating if two answers are equivalent.
Given:
- Predicted answer: {predicted}
- Ground truth answer: {ground_truth}

Evaluate if these answers are mathematically equivalent, considering:
1. Different ways of expressing the same number (e.g., "1/2" vs "0.5")
2. Different units or formats (e.g., "2 meters" vs "200 cm")
3. Different levels of precision
4. Different ways of writing equations

Respond with ONLY a number between 0 and 1, where:
- 1.0 means the answers are definitely equivalent
- 0.0 means the answers are definitely not equivalent
- Values in between indicate partial equivalence

Your score:"""

    try:
        response = llm.generate(prompt)
        score = float(''.join(c for c in response if c.isdigit() or c == '.'))
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return 0.0

def train_mcts(
    dataset: MATH500Dataset,
    llm_mix: List[LLMClient],
    num_epochs: int = 5,
    num_sub_questions: int = 2,
    num_candidate_sub_answers: int = 2,
    use_aggregator: bool = True,
    max_depth: int = 3,
    iterations: int = 100
):
    metrics = {
        "accuracy": [],
        "average_reward": [],
        "average_depth": []
    }
    
    evaluator_llm = LLMClient("mistral-8b-instruct", api_key="[api_key]")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_metrics = {
            "accuracy": [],
            "rewards": [],
            "depths": []
        }
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for idx in indices:
            sample = dataset[idx]
            print(f"\nProcessing sample {idx + 1}/{len(dataset)}")
            print(f"Problem: {sample['problem']}")
            
            try:
                root = Node(sample["problem"])
                mcts = MCTS(
                    root=root,
                    llm_mix=llm_mix,
                    num_sub_questions=num_sub_questions,
                    num_candidate_sub_answers=num_candidate_sub_answers,
                    use_aggregator=use_aggregator,
                    max_depth=max_depth,
                    iterations=iterations
                )
                
                mcts.run()
                
                result = mcts.best_action()
                if result:
                    best_action, best_state = result
                    
                    final_answer = best_state.split("\n")[-1]
                    
                    accuracy = evaluate_answer_with_llm(final_answer, sample["answer"], evaluator_llm)
                    epoch_metrics["accuracy"].append(accuracy)
                    
                    if hasattr(mcts, "answer_history") and mcts.answer_history:
                        rewards = [r for r in mcts.answer_history.values()]
                        epoch_metrics["rewards"].append(np.mean(rewards) if rewards else 0.0)
                    else:
                        epoch_metrics["rewards"].append(0.0)
                    epoch_metrics["depths"].append(mcts.max_depth)
                    
                    print(f"Predicted answer: {final_answer}")
                    print(f"Ground truth: {sample['answer']}")
                    print(f"LLM Evaluation Score: {accuracy:.2f}")
                else:
                    print("No result generated for this sample")
                    epoch_metrics["accuracy"].append(0.0)
                    epoch_metrics["rewards"].append(0.0)
                    epoch_metrics["depths"].append(0)
                    
            except Exception as e:
                print(f"Error processing sample: {e}")
                epoch_metrics["accuracy"].append(0.0)
                epoch_metrics["rewards"].append(0.0)
                epoch_metrics["depths"].append(0)
            
        metrics["accuracy"].append(np.mean(epoch_metrics["accuracy"]) if epoch_metrics["accuracy"] else 0.0)
        metrics["average_reward"].append(np.mean(epoch_metrics["rewards"]) if epoch_metrics["rewards"] else 0.0)
        metrics["average_depth"].append(np.mean(epoch_metrics["depths"]) if epoch_metrics["depths"] else 0.0)
        
        print(f"\nEpoch {epoch + 1} Metrics:")
        print(f"Average Accuracy: {metrics['accuracy'][-1]:.2f}")
        print(f"Average Reward: {metrics['average_reward'][-1]:.2f}")
        print(f"Average Depth: {metrics['average_depth'][-1]:.2f}")
    
    return metrics

def main():
    llm_mix = [
        LLMClient("llama-3.1-8b-instruct", api_key="[api_key]"),
        LLMClient("qwen-2-7b-instruct", api_key="[api_key]"),
        LLMClient("mistral-8b-instruct", api_key="[api_key]"),
        LLMClient("glm-4-9b-chat", api_key="[api_key]")
    ]
    
    dataset = MATH500Dataset(
        data_path="test.jsonl",
        split="train",
        max_samples=3
    )
    
    metrics = train_mcts(
        dataset=dataset,
        llm_mix=llm_mix,
        num_epochs=2,
        num_sub_questions=2,
        num_candidate_sub_answers=2,
        use_aggregator=True,
        max_depth=3,
        iterations=10
    )
    
    with open("mcts_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining completed!")
    print("Metrics saved to mcts_metrics.json")

if __name__ == "__main__":
    main()
