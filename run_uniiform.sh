

for norm in {"L2","Linf"}
do
	for dataset in {"SUN",}
	do
		for lamnorm in `seq -7 2 1`
		do
				CUDA_VISIBLE_DEVICES=4 python optimization_attack_baseline_two.py   --dataset $dataset   --norm $norm   --lamnorm $lamnorm    --confidence=8
				CUDA_VISIBLE_DEVICES=4 python optimization_attack_baseline_two.py   --dataset $dataset   --norm $norm   --lamnorm $lamnorm    --confidence=2
		done
	done
done