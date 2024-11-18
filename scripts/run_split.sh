# Please run this script from the root of the repository
python scripts/split_data.py --dataset_csv example_data/random_medical_data.csv --train 0.8 --val 0.1 --test 0.1 --seed 0 --output_dir example_data/ --columns "Source_LB,Source_RB,Cancer_LB,Cancer_RB" --overwrite
