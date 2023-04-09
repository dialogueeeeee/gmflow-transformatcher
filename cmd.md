## chairs
''' shell
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py \
--launcher pytorch \
--checkpoint_dir checkpoints/chairs-gmflow \
--batch_size 16 \
--val_dataset chairs sintel kitti \
--lr 4e-4 \
--image_size 384 512 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
2>&1 | tee -a checkpoints/chairs-gmflow/train.log \
&


nohup python -m torch.distributed.launch --nproc_per_node=4 main.py --launcher pytorch --checkpoint_dir checkpoints/chairs-gmflow --batch_size 16 --val_dataset chairs sintel kitti --lr 4e-4 --image_size 384 512 --padding_factor 16 --upsample_factor 8 --with_speed_metric --val_freq 10000 --save_ckpt_freq 10000 --num_steps 100000 2>&1 | tee -a checkpoints/chairs-gmflow/train.log &

nohup python -m torch.distributed.launch --nproc_per_node=4 main.py --launcher pytorch --checkpoint_dir checkpoints/chairs-gmflow --batch_size 16 --val_dataset chairs sintel kitti --lr 4e-4 --image_size 384 512 --padding_factor 16 --upsample_factor 8 --with_speed_metric --val_freq 10000 --save_ckpt_freq 10000 --num_steps 100000 >trans01.log 2>&1 &

python main.py --checkpoint_dir checkpoints/chairs-gmflow --batch_size 4 --val_dataset chairs sintel kitti --lr 4e-4 --image_size 384 512 --padding_factor 16 --upsample_factor 8 --with_speed_metric --val_freq 10000 --save_ckpt_freq 10000 --num_steps 100000
'''