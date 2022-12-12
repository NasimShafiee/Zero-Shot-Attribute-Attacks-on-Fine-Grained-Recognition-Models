

for norm in {"L2",}
do
		
	CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_test_ablation2.py   --dataset="CUB"   --norm="L2"  --lamnorm=-1   --lam4=0  --confidence=8
	CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_test_ablation2.py   --dataset="SUN"   --norm="L2"   --lamnorm=-5   --lam4=0  --confidence=8
	CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_test_ablation2.py   --dataset="AWA2"  --norm="L2"   --lamnorm=-3   --lam4=0  --confidence=8

	CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_test_ablation2.py   --dataset="CUB"   --norm="Linf"  --lamnorm=-5   --lam4=0  --confidence=8
	CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_test_ablation2.py   --dataset="SUN"   --norm="Linf"   --lamnorm=-1   --lam4=0  --confidence=8
	CUDA_VISIBLE_DEVICES=5 python optimization_attack_dazlepp_test_ablation2.py   --dataset="AWA2"  --norm="Linf"   --lamnorm=-3   --lam4=0  --confidence=8
done