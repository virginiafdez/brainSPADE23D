python ../train.py --name BRAINSPADEV3_25_nomodisc_nodiffs_selfs --batchSize 16 --gpu_ids 0 --beta1 0.5 --beta2 0.99 --lambda_vgg 1.5 \
--lambda_feat 0.25 --label_dir /media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/labels_train --image_dir \
/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/images_train --label_dir_val \
/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/labels_validation --image_dir_val \
/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/images_validation \
--cut a --label_nc 12 --nef 16 --ngf 16 --ndf 64 --use_vae --output_nc 1 --niter 200 --niter_decay 50 --checkpoints_dir \
/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION --save_epoch_freq 1 --mod_disc_epoch 25 --train_enc_only 20000 \
--display_enc_freq 7000 --display_freq 500 --disp_D_freq 2500 --print_freq 250 --z_dim 16 --batch_acc_fr 4 --lr 0.0002 \
--disc_acc_upperth 0.75 --disc_acc_lowerth 0.6 --sequences T1 FLAIR T2 --lambda_kld 0.0005 --new_size 256 256 \
--datasets SABRE ADNI BRATS-TCIA --nThreads 0 --mod_disc --lambda_mdmod 0.75 --lambda_mddat 0.3 --intensity_transform \
True --print_grad_freq 50 --steps_accuracy 15 --norm_G spectralspadeinstance3x3 --self_supervised_training 1.0 \
--distance_metric cosim --save_epoch_copy 50 --gradients_freq 4000 --activations_freq 4000 --TTUR_factor 2 --num_D 3 \
--n_layers_D 3 --type_prior N --D_modality_class --upsampling_type upsample --dataset_type sliced --skullstrip --mod_disc_path \
/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/MODISC_22/25_net_MD.pth --topK_discrim --cache_type none --lambda_slice_consistency \
1.0 --use_tboard --format_data npz --continue_train