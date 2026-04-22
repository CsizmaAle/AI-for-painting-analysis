import functions_image_formatting as fif 

def all_data_needs_is_this():
    fif.resize_normalize_images(
        input_folder="photos", 
        output_folder="photos4", 
        size=(300, 300)
    )

    fif.split_dataset(
        input_folder= "photos4",
        output_folder="dataset_split"
    )

    fif.create_csv_split_dataset(
        input_folder="dataset_split/test",
        csv_input_file="metadata.csv",
        csv_output_file="test_data.csv"
    )

    fif.create_csv_split_dataset(
        input_folder="dataset_split/train",
        csv_input_file="metadata.csv",
        csv_output_file="train_data.csv"
    )

    