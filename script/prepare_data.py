from collections import defaultdict
from typing import Any, DefaultDict, List, Dict

from transformers.tokenization_utils import PreTrainedTokenizer

from sklearn.model_selection import train_test_split
import torch
from transformers import BatchEncoding

from data_reader import DataPoint


def extract_warning_types(data: List[DataPoint]) -> List[str]:
    all_warnings: List[str] = []
    for sample in data:
        if sample.linter_report.rule_id not in all_warnings:
            all_warnings.append(sample.linter_report.rule_id)
    return all_warnings


def filter_rule(data: List[DataPoint], rule_type: str) -> List[DataPoint]:
    filtered_data: List[DataPoint] = []
    for point in data:
        if point.linter_report.rule_id == rule_type:
            filtered_data.append(point)
    return filtered_data


def split_filtered(filtered_data: List[DataPoint], include_warning: bool, model_name: str, seed=13):
    filtered_data_temp = filtered_data

    inputs = [data_point.GetT5Representation(include_warning)[0] for data_point in filtered_data]
    outputs = [
        data_point.GetT5Representation(include_warning)[1] for data_point in filtered_data_temp
    ]

    test_size = 0.1 if len(inputs) >= 10 else 1 / len(inputs)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, outputs, shuffle=True, random_state=seed, test_size=test_size
    )

    train_info, test_info = train_test_split(
        filtered_data, shuffle=True, random_state=seed, test_size=test_size
    )
    
    val_size = 0.1 if len(train_inputs) >= 10 else 1 / len(train_inputs)
   

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_inputs, train_labels, shuffle=True, random_state=seed, test_size=val_size
    )

    train_info, val_info = train_test_split(
        train_info, shuffle=True, random_state=seed, test_size=test_size
    )

    return (
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
        test_inputs,
        test_labels,
        train_info,
        val_info,
        test_info,
    )
    
def split_filtered_kfold(filtered_data: List[DataPoint], include_warning: bool, model_name: str, k_fold: int, seed=13):
    filtered_data_temp = filtered_data

    inputs = [data_point.GetT5Representation(include_warning)[0] for data_point in filtered_data]
    outputs = [
        data_point.GetT5Representation(include_warning)[1] for data_point in filtered_data_temp
    ]
    size_of_instance = len(inputs)
    divide = int(size_of_instance / 3)
    mod = size_of_instance % 3
    
    if mod == 1:
        divide1 = divide + 1
        divide2 = divide1 + divide
    elif mod == 2:
        divide1 = divide + 1
        divide2 = divide1 + divide + 1
    else:
        divide1 = divide
        divide2 = divide1 + divide

    if k_fold == 0:
        train_inputs = inputs
        test_inputs = []
        train_labels = outputs
        test_labels = []
        train_info = filtered_data
        test_info = []
    elif k_fold == 1:
        train_inputs = inputs[divide1:]
        test_inputs = inputs[0:divide1]
        train_labels = outputs[divide1:]
        test_labels = outputs[0:divide1]
        train_info = filtered_data[divide1:]
        test_info = filtered_data[0:divide1]
    elif k_fold == 2:
        train_inputs = inputs[0:divide1] + inputs[divide2:]
        test_inputs = inputs[divide1:divide2]
        train_labels = outputs[0:divide1] + outputs[divide2:]
        test_labels = outputs[divide1:divide2]
        train_info = filtered_data[0:divide1] + filtered_data[divide2:]
        test_info = filtered_data[divide1:divide2]
    elif k_fold == 3:
        train_inputs = inputs[0:divide2]
        test_inputs = inputs[divide2:]
        train_labels = outputs[0:divide2]
        test_labels = outputs[divide2:]
        train_info = filtered_data[0:divide2]
        test_info = filtered_data[divide2:]

    # test_size = 0.1 if len(inputs) >= 10 else 1 / len(inputs)
    # train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    #     inputs, outputs, shuffle=True, random_state=seed, test_size=test_size
    # )

    # train_info, test_info = train_test_split(
    #     filtered_data, shuffle=True, random_state=seed, test_size=test_size
    # )
    # print(filtered_data[0])
    # print(train_info[0])
    # print(type(filtered_data))
    # print(type(train_info))
    val_size = 0.1 if len(train_inputs) >= 10 else 1 / len(train_inputs)
   

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_inputs, train_labels, shuffle=True, random_state=seed, test_size=val_size
    )

    train_info, val_info = train_test_split(
        train_info, shuffle=True, random_state=seed, test_size=val_size
    )

    return (
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
        test_inputs,
        test_labels,
        train_info,
        val_info,
        test_info,
    )


def create_data(
    data: List[DataPoint], linter_warnings: List[str], include_warning: bool, model_name: str, k_fold: int,
):
    train: List[str] = []
    train_labels: List[str] = []
    val: List[str] = []
    val_labels: List[str] = []

    test: DefaultDict[str, List[str]] = defaultdict(list)
    test_labels: DefaultDict[str, List[str]] = defaultdict(list)
    n_test_samples = 0

    train_info: List[DataPoint] = []
    val_info: List[DataPoint] = []
    test_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)
    i = 0
    
    for warning in linter_warnings:
        filtered_data = filter_rule(data, warning)
        if len(filtered_data) <= 1:
            continue
        # (
        #     train_w,
        #     train_w_labels,
        #     val_w,
        #     val_w_labels,
        #     test_w,
        #     test_w_labels,
        #     train_w_info,
        #     val_w_info,
        #     test_w_info,
        # ) = split_filtered(filtered_data, include_warning, model_name)
        (
            train_w,
            train_w_labels,
            val_w,
            val_w_labels,
            test_w,
            test_w_labels,
            train_w_info,
            val_w_info,
            test_w_info,
        ) = split_filtered_kfold(filtered_data, include_warning, model_name, k_fold)

        train += train_w
        train_labels += train_w_labels

        val += val_w
        val_labels += val_w_labels

        train_info += train_w_info
        val_info += val_w_info

        test[warning] = test_w
        test_labels[warning] = test_w_labels

        test_info[warning] = test_w_info

        n_test_samples += len(test_w)
    print(
        "train size: {}\nval size: {}\ntest size: {}".format(len(train), len(val), n_test_samples)
    )
    return train, train_labels, val, val_labels, test, test_labels, train_info, val_info, test_info

def create_all_test_data(
    data: List[DataPoint], linter_warnings: List[str], include_warning: bool, model_name: str
):
    train: List[str] = []
    train_labels: List[str] = []
    val: List[str] = []
    val_labels: List[str] = []

    test: DefaultDict[str, List[str]] = defaultdict(list)
    test_labels: DefaultDict[str, List[str]] = defaultdict(list)
    n_test_samples = 0

    train_info: List[DataPoint] = []
    val_info: List[DataPoint] = []
    test_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)

    for warning in linter_warnings:
        count_test = 0
        for data_point in filter_rule(data, warning):
            test[warning] += [data_point.GetT5Representation(include_warning)[
                0]]
            test_labels[warning] += [
                data_point.GetT5Representation(include_warning)[1]]
            test_info[warning] += [data_point]
            count_test += 1
        n_test_samples += count_test

    print(
        "train size: {}\nval size: {}\ntest size: {}".format(
            len(train), len(val), n_test_samples)
    )

    return train, train_labels, val, val_labels, test, test_labels, train_info, val_info, test_info

class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, targets: BatchEncoding):
        self.encodings = encodings
        self.target_encodings = targets

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.target_encodings["input_ids"][index], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def create_dataset(
    inputs: List[str],
    labels: List[str],
    tokenizer: PreTrainedTokenizer,
    pad_truncate: bool,
    max_length=None,
) -> BugFixDataset:
    if max_length is not None:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
    else:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=256
        )
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=256
        )

    dataset = BugFixDataset(input_encodings, label_encodings)
    return dataset
