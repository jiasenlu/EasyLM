python -m EasyLM.models.olmo.olmo_train \
    --mesh_dim='1,-1,4' \
    --dtype='fp32' \
    --num_epochs=2 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_olmo_config='7b' \
    --update_olmo_config='' \
    --load_dataset_state='' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.3 \
    --optimizer.accumulate_gradient_steps=1 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=False \
    --train_dataset.type='huggingface' \
    --train_dataset.huggingface_dataset.path='cnn_dailymail' \
    --train_dataset.huggingface_dataset.name '3.0.0' \
    --train_dataset.huggingface_dataset.split 'validation' \
    --train_dataset.huggingface_dataset.batch_size 1 \
    --logger.output_dir="test_output" \
    --load_checkpoint='params::OLMo-7B-eazylm' \
    # --train_dataset.type='json_torch' \
    # --train_dataset.text_processor.fields='[prompt],completion' \
    # --train_dataset.json_torch_dataset.path='alpaca.jsonl' \
    # --train_dataset.json_torch_dataset.seq_length=1024 \
    # --train_dataset.json_torch_dataset.batch_size=1 \