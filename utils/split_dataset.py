import random
random.seed(2024)
def split_dataset(file_path, N):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    samples = [(content, label) for content, label in (line.strip().split('\t') for line in lines)]
    positive_samples = [sample for sample in samples if sample[1] == '1']
    positive_samples2 = [sample for sample in samples if sample[1] == '2']
    negative_samples = [sample for sample in samples if sample[1] == '0']

    test_positive = random.sample(positive_samples, N)
    test_positive2 = random.sample(positive_samples2, N)
    test_negative = random.sample(negative_samples, N)
    test_set = test_positive + test_negative + test_positive2

    train_set = [sample for sample in samples if sample not in test_set]

    return train_set, test_set

def save_dataset(train_set, test_set, train_file, test_file):
    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(f"{content}\t{label}\n" for content, label in train_set)

    with open(test_file, 'w', encoding='utf-8') as file:
        file.writelines(f"{content}\t{label}\n" for content, label in test_set)

if __name__ == "__main__":
    file_path = 'train_allin_827V2.txt'  # 替换为你的txt文件路径
    N = 130  # 替换为你需要的正样本和负样本数量
    train_file = 'train_allin_allLLM_val925.txt'
    test_file = 'val_allin_allLLM_val925.txt'

    train_set, test_set = split_dataset(file_path, N)
    save_dataset(train_set, test_set, train_file, test_file)
    print(f"训练集已保存到 {train_file}")
    print(f"测试集已保存到 {test_file}")