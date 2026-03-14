import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Evaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.smooth = SmoothingFunction().method1

    def calculate_bleu(self, reference, hypothesis):
        """计算BLEU分数"""
        # 将字符串转换为字符列表
        ref = [list(reference)]
        hyp = list(hypothesis)

        # 计算BLEU
        bleu = sentence_bleu(ref, hyp, weights=(0.5, 0.5), smoothing_function=self.smooth)
        return bleu

    def evaluate_generation(self, test_pairs):
        """评估生成结果"""
        self.model.eval()
        results = []
        bleu_scores = []

        with torch.no_grad():
            for input_text, target_text in test_pairs:
                # 编码输入
                input_ids = self.tokenizer.encode(input_text, self.config.max_length)
                src = torch.tensor([input_ids]).to(self.config.device)

                # 生成
                generated_ids = self.model.generate(src, self.config.max_length)
                generated_text = self.tokenizer.decode(generated_ids[0].cpu().numpy())

                # 计算BLEU
                bleu = self.calculate_bleu(target_text, generated_text)
                bleu_scores.append(bleu)

                results.append({
                    'input': input_text,
                    'target': target_text,
                    'generated': generated_text,
                    'bleu': bleu
                })

        # 统计
        avg_bleu = np.mean(bleu_scores)

        return results, avg_bleu

    def print_examples(self, results, n=5):
        """打印生成示例"""
        print("\n生成示例：")
        print("=" * 60)

        for i, result in enumerate(results[:n]):
            print(f"\n示例 {i + 1}:")
            print(f"上联: {result['input']}")
            print(f"下联: {result['target']}")
            print(f"生成: {result['generated']}")
            print(f"BLEU: {result['bleu']:.4f}")
            print("-" * 40)