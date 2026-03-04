import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMOpponent:
    def __init__(self, model_name="distilgpt2"):
        print("🧠 Loading LLM opponent...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def state_to_text(self, state):
        """
        Convert environment state → text prompt
        """
        return f"Game state: {state}. Choose best action from [0,1,2,3,4]. Action:"

    def get_action(self, state):
        prompt = self.state_to_text(state)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 5,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )

        generated_text = self.tokenizer.decode(outputs[0])

        # Extract action from generated text
        for i in range(5):
            if str(i) in generated_text[-10:]:
                return i

        # fallback
        return torch.randint(0, 5, (1,)).item()