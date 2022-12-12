
for norm in {"Linf",}
do
	for dataset in {"SUN","CUB","AWA2"}
	do
		CUDA_VISIBLE_DEVICES=4 python optimization_attack_dazlepp_test_split_projectLinf.py   --dataset $dataset   --norm $norm   --lam4=0 --confidence=8
		# for lam4 in `seq 1 -2 -7`
		# do
			# CUDA_VISIBLE_DEVICES=4 python optimization_attack_dazlepp_test_split_projectLinf.py   --dataset $dataset   --norm $norm   --lam4 $lam4  --confidence=8
			
		# done
	done
done