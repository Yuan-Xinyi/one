"""Auto-reset distribution driven by a frozen seed-policy snapshot."""
from __future__ import annotations

import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.seed_deployment import (
    SeedDeploymentConfig,
    deployment_config_from_checkpoint,
    select_seed_deployment,
)


class SeedPolicyLineDistribution:
    """Drop-in ``LineDistribution`` that decouples task and initial q.

    The task is sampled from a cached `(p0, line_dir, n_target, candidates)`
    dataset. A frozen seed policy chooses q0. Exploration mixes policy,
    uniform-valid and fallback selections during controller adaptation to
    prevent seed collapse from narrowing the controller's reset distribution.
    """

    def __init__(
        self,
        dataset: CachedSeedCandidateDataset,
        seed_policy,
        kin,
        *,
        policy_prob: float = 0.7,
        uniform_prob: float = 0.2,
        fallback_prob: float = 0.1,
        deterministic_policy: bool = False,
        include_log_manip: bool = False,
        include_ray_error: bool = False,
        include_directional_dynamics: bool = False,
        independent_rng_streams: bool = True,
        seed: int = 0,
        seed_deployment: SeedDeploymentConfig | dict | None = None,
    ):
        probs = torch.tensor(
            [policy_prob, uniform_prob, fallback_prob], dtype=torch.float32)
        if bool((probs < 0).any().item()) or not torch.isclose(
                probs.sum(), torch.tensor(1.0), atol=1e-6):
            raise ValueError('policy/uniform/fallback probabilities must be non-negative and sum to 1')
        self.dataset = dataset
        self.seed_policy = seed_policy
        self.kin = kin
        self.policy_prob = float(policy_prob)
        self.uniform_prob = float(uniform_prob)
        self.fallback_prob = float(fallback_prob)
        self.deterministic_policy = bool(deterministic_policy)
        self.include_log_manip = bool(include_log_manip)
        self.include_ray_error = bool(include_ray_error)
        self.include_directional_dynamics = bool(
            include_directional_dynamics)
        if seed_deployment is None:
            self.seed_deployment = SeedDeploymentConfig()
        elif isinstance(seed_deployment, SeedDeploymentConfig):
            self.seed_deployment = seed_deployment
        elif isinstance(seed_deployment, dict):
            self.seed_deployment = deployment_config_from_checkpoint(
                {'seed_deployment': seed_deployment})
        else:
            raise TypeError(
                'seed_deployment must be a SeedDeploymentConfig, dict, or None')
        self.cpu_generator = torch.Generator().manual_seed(int(seed))
        self.independent_rng_streams = bool(independent_rng_streams)
        if self.independent_rng_streams:
            self.policy_generator = torch.Generator(
                device=kin.device).manual_seed(int(seed) + 1)
            self.uniform_generator = torch.Generator(
                device=kin.device).manual_seed(int(seed) + 2)
            self.mode_generator = torch.Generator(
                device=kin.device).manual_seed(int(seed) + 3)
        else:
            # Compatibility path for version-3 runs created before reset
            # components received independent random streams.
            shared = torch.Generator(
                device=kin.device).manual_seed(int(seed) + 1)
            self.policy_generator = shared
            self.uniform_generator = shared
            self.mode_generator = shared
        self.seed_policy.eval()

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None
               ) -> dict[str, torch.Tensor]:
        # NSRLBatchedEnv currently calls sample without a generator. Honor an
        # explicit CPU generator for compatibility with scripted callers.
        cpu_generator = self.cpu_generator if generator is None else generator
        candidates = self.dataset.sample(n, generator=cpu_generator).to(
            self.kin.device, dtype=self.kin.dtype)
        features = initial_observation_features(
            self.kin, candidates,
            include_log_manip=self.include_log_manip,
            include_ray_error=self.include_ray_error,
            include_directional_dynamics=(
                self.include_directional_dynamics))
        dist, _, feasibility = self.seed_policy.distribution_and_values(
            features, candidates.valid)
        if self.seed_deployment.mode == 'conservative':
            policy_index = select_seed_deployment(
                dist.logits, feasibility, candidates.valid,
                self.seed_deployment).selected_index
        elif self.deterministic_policy:
            policy_index = dist.logits.argmax(dim=-1)
        else:
            policy_index = torch.multinomial(
                dist.probs, 1, generator=self.policy_generator).squeeze(-1)
        uniform_index = torch.multinomial(
            candidates.valid.float(), 1,
            generator=self.uniform_generator).squeeze(-1)
        first_valid_index = candidates.valid.float().argmax(dim=-1)

        # The native/pilot action is explicit dataset metadata. A custom cache
        # without one uses its first valid action as the deterministic fallback,
        # matching the evaluation baseline; uniform exploration is a separate
        # reset component above.
        if self.dataset.fallback_index is None:
            fallback_index = first_valid_index
        else:
            fallback_index = torch.full_like(
                policy_index, self.dataset.fallback_index)
            row = torch.arange(n, device=candidates.device)
            fallback_index = torch.where(
                candidates.valid[row, fallback_index],
                fallback_index, first_valid_index)

        mode = torch.rand(n, device=candidates.device,
                          generator=self.mode_generator)
        selected_index = torch.where(
            mode < self.policy_prob,
            policy_index,
            torch.where(
                mode < self.policy_prob + self.uniform_prob,
                uniform_index,
                fallback_index,
            ),
        )
        selected = candidates.select(selected_index)
        return selected.specs()
