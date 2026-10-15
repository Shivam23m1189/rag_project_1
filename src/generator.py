from transformers import pipeline

class QwenGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.llm = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            clean_up_tokenization_spaces=False,
        )

    def generate(self, query: str, context_chunks: list[str], max_new_tokens: int = 150) -> str:
        context_str = " --- ".join(context_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                    "If the answer is not in the context, say 'I cannot find the answer in the provided documents.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}",
            },
        ]

        prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.llm(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=True,
        )
        generated_text = response[0]["generated_text"]
        return generated_text[len(prompt):].strip()