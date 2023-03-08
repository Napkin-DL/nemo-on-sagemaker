# Manifest Utils
from tqdm.auto import tqdm
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import unicodedata
# Preprocessing steps
import re
import unicodedata
import pickle


CHARS_TO_IGNORE_REGEX = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\{\}\【\】\・\。\『\』\、\ー\〜]'  # remove special character tokens
MAX_COUNT = 32
BASE_INPUT_DIR = "/opt/ml/processing/input"
BASE_OUTPUT_DIR = "/opt/ml/processing/output"
UNCOMMON_TOKENS_COUNT = 5


def remove_special_characters(data):
    data["text"] = re.sub(CHARS_TO_IGNORE_REGEX, '', data["text"]).lower().strip()
    return data


def change_filepath(data):
    path_list = data['audio_filepath'].split('/')
    if 'train' in path_list:
        src_path = '/Users/choijoon/workspace/external-package/NeMo/NeMo/datasets/ko/train/wav'
        des_path = f'{BASE_INPUT_DIR}/ko/train/wav'
    elif 'dev' in path_list:
        src_path = '/Users/choijoon/workspace/external-package/NeMo/NeMo/datasets/ko/dev/wav'
        des_path = f'{BASE_INPUT_DIR}/ko/dev/wav'
    elif 'test' in path_list:
        src_path = '/Users/choijoon/workspace/external-package/NeMo/NeMo/datasets/ko/test/wav'
        des_path = f'{BASE_INPUT_DIR}/ko/test/wav'

    data['audio_filepath'] = data['audio_filepath'].replace(src_path, des_path)
    print(f"data : {data}")
    return data


def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = BASE_OUTPUT_DIR + '/cleaned'
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir, exist_ok=True)
        
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath

# Processing pipeline
def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest



def main():
    train_manifest = f"{BASE_INPUT_DIR}/manifests/commonvoice_train_manifest.json"
    dev_manifest = f"{BASE_INPUT_DIR}/manifests/commonvoice_dev_manifest.json"
    test_manifest = f"{BASE_INPUT_DIR}/manifests/commonvoice_test_manifest.json"
    
    train_manifest_data = read_manifest(train_manifest)
    dev_manifest_data = read_manifest(dev_manifest)
    test_manifest_data = read_manifest(test_manifest)
    
    train_text = [data['text'] for data in train_manifest_data]
    dev_text = [data['text'] for data in dev_manifest_data]
    test_text = [data['text'] for data in test_manifest_data]
    
    train_charset = get_charset(train_manifest_data)
    dev_charset = get_charset(dev_manifest_data)
    test_charset = get_charset(test_manifest_data)
    
    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))
    test_set = set(test_charset.keys())
    
    print(f"Number of tokens in train+dev set : {len(train_dev_set)}")
    print(f"Number of tokens in test set : {len(test_set)}")
    
    
    # OOV tokens in test set
    train_test_common = set.intersection(train_dev_set, test_set)
    test_oov = test_set - train_test_common
    print(f"Number of OOV tokens in test set : {len(test_oov)}")
    print()
    print(test_oov)

    
    # Populate dictionary mapping count: list[tokens]
    train_counts = defaultdict(list)
    for token, count in train_charset.items():
        train_counts[count].append(token)
    for token, count in dev_charset.items():
        train_counts[count].append(token)

    # Compute sorter order of the count keys
    count_keys = sorted(list(train_counts.keys()))
    

    TOKEN_COUNT_X = []
    NUM_TOKENS_Y = []
    for count in range(1, MAX_COUNT + 1):
        if count in train_counts:
            num_tokens = len(train_counts[count])

            TOKEN_COUNT_X.append(count)
            NUM_TOKENS_Y.append(num_tokens)
            
            
    

    plt.bar(x=TOKEN_COUNT_X, height=NUM_TOKENS_Y)
    plt.title("Occurance of unique tokens in train+dev set")
    plt.xlabel("# of occurances")
    plt.ylabel("# of tokens")
    plt.xlim(0, MAX_COUNT);
    plt.savefig(f'{BASE_OUTPUT_DIR}/occurance_tokens.pdf')
    
    

    chars_with_infrequent_occurance = set()
    for count in range(1, UNCOMMON_TOKENS_COUNT + 1):
        if count in train_counts:
            token_list = train_counts[count]
            chars_with_infrequent_occurance.update(set(token_list))

    print(f"Number of tokens with <= {UNCOMMON_TOKENS_COUNT} occurances : {len(chars_with_infrequent_occurance)}")
    
    all_tokens = set.union(train_dev_set, test_set)
    print(f"Original train+dev+test vocab size : {len(all_tokens)}")

    extra_char = set(test_oov)
    train_token_set = all_tokens - extra_char
    print(f"New train vocab size : {len(train_token_set)}")
    
    
    # if PERFORM_DAKUTEN_NORMALIZATION:
    #     normalized_train_token_set = set()
    #     for token in train_token_set:
    #         normalized_token = process_dakuten(str(token))
    #         normalized_train_token_set.update(normalized_token)

    #     print(f"After dakuten normalization, number of train tokens : {len(normalized_train_token_set)}")
    # else:
    normalized_train_token_set = train_token_set
    

    # Load manifests
    train_data = read_manifest(train_manifest)
    dev_data = read_manifest(dev_manifest)
    test_data = read_manifest(test_manifest)
    
        # List of pre-processing functions
    PREPROCESSORS = [
        remove_special_characters,
        # remove_extra_kanji,
        # remove_dakuten,
        change_filepath,
    ]

    # Apply preprocessing
    train_data_processed = apply_preprocessors(train_data, PREPROCESSORS)
    dev_data_processed = apply_preprocessors(dev_data, PREPROCESSORS)
    test_data_processed = apply_preprocessors(test_data, PREPROCESSORS)

    # Write new manifests
    train_manifest_cleaned = write_processed_manifest(train_data_processed, train_manifest)
    dev_manifest_cleaned = write_processed_manifest(dev_data_processed, dev_manifest)
    test_manifest_cleaned = write_processed_manifest(test_data_processed, test_manifest)
    
    train_manifest_data = read_manifest(train_manifest_cleaned)
    train_charset = get_charset(train_manifest_data)

    dev_manifest_data = read_manifest(dev_manifest_cleaned)
    dev_charset = get_charset(dev_manifest_data)

    train_dev_set = set.union(set(train_charset.keys()), set(dev_charset.keys()))

    
    print(f"Number of tokens in preprocessed train+dev set : {len(train_dev_set)}")
    
    
    with open(f'{BASE_OUTPUT_DIR}/commonvoice_ko.pkl', 'wb') as f:
        pickle.dump(list(train_dev_set),f)
        

if __name__ == "__main__":
    main()