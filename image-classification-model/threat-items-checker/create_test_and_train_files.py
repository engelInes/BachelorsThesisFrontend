import os


def generate_txt_files_from_ranges(train_start, train_end, test_start, test_end, train_output_path, test_output_path):
    with open(train_output_path, 'w') as train_file:
        for number in range(train_start, train_end + 1):
            train_file.write(f"E:/OPIXray/train/train_image/{number:06d}.jpg\n")

    with open(test_output_path, 'w') as test_file:
        for number in range(test_start, test_end + 1):
            test_file.write(f"E:/OPIXray/test/test_image/{number:06d}.jpg\n")

    print(f"Created {train_output_path} with {train_end - train_start + 1} training images")
    print(f"Created {test_output_path} with {test_end - test_start + 1} test images")


if __name__ == "__main__":
    train_start_num = 9000
    train_end_num = 42398

    test_start_num = 10466
    test_end_num = 42996

    train_txt_path = "E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/content/darknet/data/train.txt"
    test_txt_path = "E:/facultate/licenta/implementation/diagramsGit/BachelorsThesisFrontend/image-classification-model/content/darknet/data/test.txt"

    os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_txt_path), exist_ok=True)

    generate_txt_files_from_ranges(train_start_num, train_end_num, test_start_num, test_end_num, train_txt_path,
                                   test_txt_path)