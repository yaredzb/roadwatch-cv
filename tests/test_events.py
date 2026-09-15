"""Unit tests for event lifecycle management, candidate activation, and cooldowns."""

from roadwatch.config import EventConfig
from roadwatch.events import EventManager, EventState
from roadwatch.risk_engine import RiskAssessment
from roadwatch.types import RiskLevel


def test_event_lifecycle_and_cooldown():
    """Verify Candidate -> Active -> Resolved transitions and cooldown period enforcement."""
    cfg = EventConfig(min_persistence_sec=0.5, cooldown_sec=2.0)
    em = EventManager(cfg)

    assessment = RiskAssessment(
        pedestrian_id=1,
        vehicle_id=2,
        score=4,
        risk_level=RiskLevel.HIGH,
        triggered_rules=["close_proximity", "closing_distance"],
        normalized_distance=0.04,
        shared_zones=["crosswalk"],
    )

    # Time 0.0s: First observation -> Candidate created
    em.update([assessment], timestamp=0.0)
    pair = (1, 2)
    assert pair in em.active_events
    assert em.active_events[pair].state == EventState.CANDIDATE

    # Time 0.3s: Insufficient persistence (< 0.5s) -> still Candidate
    em.update([assessment], timestamp=0.3)
    assert em.active_events[pair].state == EventState.CANDIDATE

    # Time 0.6s: Exceeded 0.5s persistence -> transitions to ACTIVE
    activated, _ = em.update([assessment], timestamp=0.6)
    assert em.active_events[pair].state == EventState.ACTIVE

    # Time 1.0s: Condition clears (no assessments) -> transitions to RESOLVED and archived
    _, resolved = em.update([], timestamp=1.0)
    assert len(resolved) == 1
    assert resolved[0].state == EventState.RESOLVED
    assert pair not in em.active_events
    assert len(em.completed_events) == 1

    # Time 1.5s: Same condition triggers again within cooldown (< 2.0s cooldown) -> ignored
    em.update([assessment], timestamp=1.5)
    assert pair not in em.active_events
