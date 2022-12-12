

for norm in {"Linf",}
do
	for dataset in {"AWA2",}
	do
		for lam4 in `seq -7 2 1`
		do
			for lamnorm in `seq -7 2 1`
			do
				CUDA_VISIBLE_DEVICES=1 python optimization_attack_dazlepp_test_split.py   --dataset $dataset   --norm $norm   --lamnorm $lamnorm   --lam4 $lam4  --confidence=2
			done
		done
	done
done