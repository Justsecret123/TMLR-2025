python -m generate_csv_multi -input results_GCN_multi.txt -output results_GCN_multi.csv &  \
python -m generate_csv_multi -input results_graphsage_multi.txt -output results_graphsage_multi.csv & \
python -m generate_csv_multi -input results_vanilla_multi.txt -output results_vanilla_multi.csv & \
python -m generate_csv_multi_score -input scores_GCN_multi.txt -output scores_GCN_multi.csv -model gcn & \
python -m generate_csv_multi_score -input scores_graphsage_multi.txt -output scores_graphsage_multi.csv -model sage
