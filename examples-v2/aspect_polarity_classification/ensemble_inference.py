import random

import findfile
import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report
from pyabsa import AspectPolarityClassification as APC
from pyabsa.utils import VoteEnsemblePredictor
import warnings

warnings.filterwarnings("ignore")


def ensemble_predict(apc_classifiers: dict, text, print_result=False):
    result = []
    for key, apc_classifier in apc_classifiers.items():
        result += apc_classifier.predict(text, print_result=print_result)["sentiment"]
    return max(set(result), key=result.count)


def ensemble_performance(dataset, print_result=False):
    ckpts = findfile.find_cwd_dirs(dataset + "_acc")
    random.shuffle(ckpts)
    apc_classifiers = {}
    for ckpt in ckpts[:]:
        apc_classifiers[ckpt] = APC.SentimentClassifier(ckpt)
    inference_file = {}

    pred = []
    gold = []
    texts = open(inference_file[dataset], "r").readlines()
    for i, text in enumerate(tqdm.tqdm(texts)):
        result = ensemble_predict(apc_classifiers, text, print_result)
        pred.append(result)
        gold.append(text.split("$LABEL$")[-1].strip())
    print(classification_report(gold, pred, digits=4))


if __name__ == "__main__":
    # Training the models before ensemble inference, take Laptop14 as an example

    for dataset in [
        APC.APCDatasetList.Laptop14,
        APC.APCDatasetList.Restaurant14,
        APC.APCDatasetList.Restaurant15,
        APC.APCDatasetList.Restaurant16,
        APC.APCDatasetList.MAMS,
    ]:
        # Training
        pass
    # Ensemble inference
    dataset_file_dict = {
        # 'laptop14': findfile.find_cwd_files(['laptop14', '.inference'], exclude_key=[]),
        "laptop14": "Laptops_Test_Gold.xml.seg.inference",
        "restaurant14": "Restaurants_Test_Gold.xml.seg.inference",
        "restaurant15": "restaurant_test.raw.inference",
        "restaurant16": "restaurant_test.raw.inference",
        "twitter": "twitter_test.raw.inference",
        "mams": "test.xml.dat.inference",
    }

    checkpoints = {
        ckpt: APC.SentimentClassifier(checkpoint=ckpt)
        for ckpt in findfile.find_cwd_dirs(or_key=["laptop14_acc"])
    }

    ensemble_predictor = VoteEnsemblePredictor(
        checkpoints, weights=None, numeric_agg="mean", str_agg="max_vote"
    )

    for key, files in dataset_file_dict.items():
        text_classifiers = {}

        print(f"Ensemble inference")
        lines = []
        if isinstance(files, str):
            files = [files]
            for file in files:
                with open(file, "r") as f:
                    lines.extend(f.readlines())

        # 测试总体准确率 batch predict
        # eval acc
        count1 = 0
        accuracy = 0
        batch_pred = []
        batch_gold = []

        # do not merge the same sentence
        results = ensemble_predictor.batch_predict(
            lines, ignore_error=False, print_result=False
        )
        it = tqdm.tqdm(results, ncols=100)
        for i, result in enumerate(it):
            label = result["sentiment"]
            if label == lines[i].split("$LABEL$")[-1].strip():
                count1 += 1
            batch_pred.append(label)
            batch_gold.append(lines[i].split("$LABEL$")[-1].strip().split(","))
            accuracy = count1 / (i + 1)
            it.set_description(f"Accuracy: {accuracy:.4f}")

        print(metrics.classification_report(batch_gold, batch_pred, digits=4))
        print(f"Final accuracy: {accuracy}")
