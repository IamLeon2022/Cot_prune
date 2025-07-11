import os
import json
from tqdm import tqdm
from llmlingua import PromptCompressor


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def compress_cot_list(data, cot_field='cot', model_path='llmlingua-2-llama-2-7b', compression_ratio=0.5):
    compressor = PromptCompressor(model_name=model_path, use_llmlingua2=True)
    compressed_data = []
    for item in tqdm(data, desc=f"Compressing with γ={compression_ratio}"):
        cot_text = item.get(cot_field, "")
        try:
            result = compressor.compress_prompt(cot_text, rate=compression_ratio)
            item['compressed_cot'] = result['compressed_prompt']
            item['original_cot_tokens'] = result['origin_tokens']
            item['compressed_cot_tokens'] = result['compressed_tokens']
            item['compression_rate'] = result['rate']
            compressed_data.append(item)
        except Exception as e:
            print(f"[Skip] Error on item {item.get('id', '')}: {e}")
    return compressed_data

def batch_compress(input_path, output_dir, cot_field='cot', model_path='llmlingua-2-llama-2-7b', ratio_list=[0.9, 0.7, 0.5]):
    data = load_jsonl(input_path)
    for ratio in ratio_list:
        output_path = os.path.join(output_dir, f"compressed_cot_{int(ratio*100)}.jsonl")
        compressed = compress_cot_list(data, cot_field=cot_field, model_path=model_path, compression_ratio=ratio)
        save_jsonl(compressed, output_path)
        avg_rate = sum(d['compression_rate'] for d in compressed) / len(compressed)
        print(f"[γ={ratio}] Saved {len(compressed)} samples → Avg compression rate: {avg_rate:.3f}")

if __name__ == "__main__":
    # 示例用法（你可以放到自己的 runner 或 notebook 中使用）
    batch_compress(
        input_path="UltraInteract_sft_sample/UltraInteract_sample10.json",
        output_dir="compressed",
        cot_field="response",  # 或 "output", "rationale"，视你的字段而定
        model_path="llmlingua-2-xlm-roberta-large-meetingbank",  # 本地模型路径
        ratio_list=[0.9, 0.8,0.7,0.6, 0.5]
    )
