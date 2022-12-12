

for norm in {"Linf",}
do
	for dataset in {"SUN",}
	do
		for lamnorm in `seq 1 -2 -7`
		do
				CUDA_VISIBLE_DEVICES=5 python optimization_attack_baseline_one.py   --dataset $dataset   --norm $norm   --lamnorm $lamnorm    --confidence=8
				CUDA_VISIBLE_DEVICES=5 python optimization_attack_baseline_one.py   --dataset $dataset   --norm $norm   --lamnorm $lamnorm    --confidence=2
		done
	done
done