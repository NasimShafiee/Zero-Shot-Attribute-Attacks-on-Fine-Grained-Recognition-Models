

for norm in {"L2",}
do
	for dataset in {"CUB",}
	do
		CUDA_VISIBLE_DEVICES=1 python optimization_attack_dazlepp_test2.py --dataset $dataset --norm $norm
	done
done