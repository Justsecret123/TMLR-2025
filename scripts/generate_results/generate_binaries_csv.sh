python -m generate_csv_binary -input results_GCN_binary.txt -output results_GCN_binary.csv &  \
python -m generate_csv_binary -input results_graphsage_binary.txt -output results_graphsage_binary.csv & \
python -m generate_csv_binary -input results_vanilla_binary.txt -output results_vanilla_binary.csv & \
python -m generate_csv_binary_score -input scores_GCN_binary.txt -output scores_GCN_binary.csv -model gcn & \
python -m generate_csv_binary_score -input scores_graphsage_binary.txt -output scores_graphsage_binary.csv -model sage