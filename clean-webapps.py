from collections import Counter

from model import get_source_text


def get_labels(dataset: str = "webapps") -> dict:
    """ Initialize data dictionary with train_labels and test_labels
    """
    data = dict()
    for dataset_type in ["test", "train"]:
        data.update(
            {f"{dataset_type}_labels": get_source_text(dataset_type=dataset_type, dataset=dataset, labels=True)})
    return data


def clean_data(data: list) -> list:
    # Define the list of valid categories
    valid_categories = ['Find Alternative', 'Filter Spam', 'Sync Accounts', 'Delete Account']

    # Iterate through the list and check each element
    for i in range(len(data)):
        if data[i] not in valid_categories:
            data[i] = 'Other'
    return data


def count_categories(data: list):
    # Count the occurrences of each category
    category_counts = Counter(data)

    for category, count in category_counts.items():
        if count > 2:
            print(f'{category}: {count}')


labels = get_labels()
print(labels)

count_categories(labels["train_labels"])
print("######")
count_categories(labels["test_labels"])

# keepers: 'Find Alternative', 'Filter Spam', 'Sync Accounts', 'Delete Account'

labels["train_labels"] = clean_data(labels["train_labels"])
labels["test_labels"] = clean_data(labels["test_labels"])

for dataset_type in ["test", "train"]:
    with open(f"webapps_{dataset_type}_ans.txt", "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in labels[f"{dataset_type}_labels"])
