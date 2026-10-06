"""Unchanged Adam PPO settings; optional research runners are omitted."""
from mjlab.rl import RslRlModelCfg
from booster_mjlab.rl.config import RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg


def booster_k1_ppo_runner_cfg(use_muon: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster K1 velocity task.

    ``use_muon=True`` is unsupported in the minimized course snapshot.
    """
    if use_muon:
        raise ValueError("Course snapshot supports the baseline Adam PPO only")
    algorithm_cls = RslRlPpoAlgorithmCfg
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=algorithm_cls(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            # only used to calculate metrics
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=False,
                use_mirror_loss=False,
                data_augmentation_func="booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries",
            ),
        ),
        experiment_name="k1_velocity" + ("_muon" if use_muon else ""),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
