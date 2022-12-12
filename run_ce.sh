

for norm in {"Linf","L2"}
do
	for dataset in {"AWA2","CUB","SUN"}
	do
		for lamnorm in `seq -7 2 1`
		do
			for lam4 in `seq -7 2 1`
			do
				CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_CrossEntropy.py   --dataset $dataset   --norm $norm   --lamnorm $lamnorm    --lam4=$lam4
			done
		done
	done
done