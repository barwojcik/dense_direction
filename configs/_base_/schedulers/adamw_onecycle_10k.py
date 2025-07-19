train_iters = 10_000
learning_rate = 0.001
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=learning_rate,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
)
param_scheduler = [
    dict(
        type="OneCycleLR",
        eta_max=learning_rate,
        total_steps=train_iters,
        by_epoch=False,
    )
]
train_cfg = dict(
    by_epoch=False,
    max_iters=train_iters,
    val_interval=1_000,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")