import os
import copy
import pathlib
import logging
import json
import torch
import tarfile

from tqdm.auto import tqdm
from omegaconf import open_dict
from nemo.collections.asr.models import EncDecCTCModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

TEST_MANIFEST_PATH = os.environ['TEST_MANIFEST_PATH']
WAV_PATH = os.environ['WAV_PATH']


def find_checkpoint(model_dir):
    checkpoint_path = None
    for (root, dirs, files) in os.walk(model_dir):
        if len(files) > 0:
            for file_name in files:
                if file_name.endswith('last.ckpt'):
                    checkpoint_path = root + '/' + file_name
    return checkpoint_path


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

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath


def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest


def change_dir(data):
    data['audio_filepath'] = data['audio_filepath'].replace(TEST_MANIFEST_PATH, WAV_PATH)
    return data


def predict(asr_model, predictions, targets, target_lengths, predictions_lengths=None):
    references = []
    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = targets.long().cpu()
        tgt_lenths_cpu_tensor = target_lengths.long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = asr_model.decoding.decode_tokens_to_str(target)
            references.append(reference)

        hypotheses, _ = asr_model.decoding.ctc_decoder_predictions_tensor(
            predictions, predictions_lengths, fold_consecutive=True
        )
    return references[0], hypotheses[0]

    
def main():
    model_path = "/opt/ml/processing/model/model.tar.gz"
    model_dir = 'trained_model'
    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_dir)
        
    
    
    logger.debug("Loading nemo model.")
    checkpoint_path = find_checkpoint(model_dir)
    print(f"checkpoint_path : {checkpoint_path}")

    asr_model = EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    asr_model = asr_model.to(device)
    asr_model.eval()
        
    logger.debug("Reading test data.")    

    test_manifest_filename = "/opt/ml/processing/input/manifest/test_manifest.json"
    eval_test_manifest_filename = "/opt/ml/processing/evaluation/eval_test_manifest.json"
    
    test_data = read_manifest(test_manifest_filename)
    test_data_processed = apply_preprocessors(test_data, [change_dir])
    local_test_manifest_path = write_processed_manifest(test_data_processed, eval_test_manifest_filename)
   
    cfg = copy.deepcopy(asr_model.cfg)
    
    with open_dict(cfg):
        cfg.test_ds.manifest_filepath = local_test_manifest_path
        
    asr_model.setup_multiple_test_data(cfg.test_ds)
    
    wer_nums = []
    wer_denoms = []
    reference_list = []
    predicted_list = []
    
    for test_batch in asr_model.test_dataloader():
        test_batch = [x.to(device) for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]
        
        log_probs, encoded_len, greedy_predictions = asr_model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )
        # Notice the model has a helper object to compute WER
        asr_model._wer.update(greedy_predictions, targets, targets_lengths)
        reference, predicted = predict(asr_model, greedy_predictions, targets, targets_lengths)
        
        reference_list.append(reference)
        predicted_list.append(predicted)
        
        _, wer_num, wer_denom = asr_model._wer.compute()
        asr_model._wer.reset()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

        # Release tensors from GPU memory
        del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

    # We need to sum all numerators and denominators first. Then divide.
    wer_result = sum(wer_nums)/sum(wer_denoms)
    
    report_dict = {
        "metrics": {
            "wer": {
                "value": wer_result
            },
        },
        "reference": {
            "value": reference_list
        },
        "predicted": {
            "value": predicted_list
        }
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with wer: %f", wer_result)
    evaluation_path = f"{output_dir}/evaluation.json"
    print ("evaluation_path", evaluation_path)
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))


if __name__ == "__main__":    
        
    main()