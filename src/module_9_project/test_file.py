from module_9_project.data import get_dataset, dataset_split, dataset_train_test_split


def foo():
    features, target = dataset_split("tests/test_data/test_data.csv")
    print(features.shape)
    print(target.shape)
    # features_train, features_val, target_train, target_val = dataset_train_test_split(
    #     file_path="tests/test_data/test_data.csv",
    #     random_state=42,
    #     test_split_ratio=0.2
    # )
    # print(features_train.shape)
    # print(features_val.shape)
    # print(target_train.shape)
    # print(target_val.shape)


