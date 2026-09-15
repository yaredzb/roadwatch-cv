"""Event lifecycle management and alert state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from roadwatch.config import EventConfig
from roadwatch.logger import get_logger
from roadwatch.risk_engine import RiskAssessment
from roadwatch.types import RiskLevel

logger = get_logger("roadwatch.events")


class EventState(str, Enum):
    """Lifecycle states of a risk event."""
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RESOLVED = "resolved"
    COOLDOWN = "cooldown"


@dataclass
class EventRecord:
    """Documented evidence and metadata for a detected risk event."""
    event_id: str
    pedestrian_id: int
    vehicle_id: int
    start_time: float
    end_time: Optional[float] = None
    state: EventState = EventState.CANDIDATE
    highest_score: int = 0
    highest_risk_level: RiskLevel = RiskLevel.LOW
    triggered_rules: Set[str] = field(default_factory=set)
    zone: Optional[str] = None
    min_normalized_distance: float = 1.0


class EventManager:
    """Manages event lifecycles, persistence thresholds, cooldowns, and event logs."""

    def __init__(self, config: EventConfig) -> None:
        self.config = config
        self.min_persistence = config.min_persistence_sec
        self.cooldown_sec = config.cooldown_sec
        self.event_counter = 0

        # Pair key: (pedestrian_id, vehicle_id)
        self.active_events: Dict[Tuple[int, int], EventRecord] = {}
        self.completed_events: List[EventRecord] = []
        self.cooldown_tracker: Dict[Tuple[int, int], float] = {}

    def _next_event_id(self) -> str:
        self.event_counter += 1
        return f"EVT-{self.event_counter:05d}"

    def update(
        self,
        assessments: List[RiskAssessment],
        timestamp: float,
    ) -> Tuple[List[EventRecord], List[EventRecord]]:
        """
        Process current frame risk assessments.
        Returns (newly_activated_events, newly_resolved_events).
        """
        newly_activated: List[EventRecord] = []
        newly_resolved: List[EventRecord] = []
        current_pairs: Set[Tuple[int, int]] = set()

        for a in assessments:
            pair = (a.pedestrian_id, a.vehicle_id)
            current_pairs.add(pair)

            # Skip if in cooldown
            if pair in self.cooldown_tracker:
                if timestamp - self.cooldown_tracker[pair] < self.cooldown_sec:
                    continue
                else:
                    del self.cooldown_tracker[pair]

            if a.score > 0 and a.risk_level != RiskLevel.LOW:
                event = self._process_active_condition(pair, a, timestamp)
                if event and event.state == EventState.ACTIVE and event not in newly_activated:
                    # Check if it just transitioned to ACTIVE
                    if round(timestamp - event.start_time, 2) >= round(self.min_persistence, 2):
                        newly_activated.append(event)

        # Resolve pairs that are no longer triggering
        for pair in list(self.active_events.keys()):
            if pair not in current_pairs:
                resolved = self._resolve_event(pair, timestamp)
                if resolved:
                    newly_resolved.append(resolved)

        return newly_activated, newly_resolved

    def _process_active_condition(
        self,
        pair: Tuple[int, int],
        a: RiskAssessment,
        timestamp: float,
    ) -> Optional[EventRecord]:
        """Update existing candidate/active event or instantiate new candidate."""
        if pair not in self.active_events:
            event = EventRecord(
                event_id=self._next_event_id(),
                pedestrian_id=pair[0],
                vehicle_id=pair[1],
                start_time=timestamp,
                highest_score=a.score,
                highest_risk_level=a.risk_level,
                triggered_rules=set(a.triggered_rules),
                zone=a.shared_zones[0] if a.shared_zones else None,
                min_normalized_distance=a.normalized_distance,
            )
            self.active_events[pair] = event

        event = self.active_events[pair]
        event.highest_score = max(event.highest_score, a.score)
        event.triggered_rules.update(a.triggered_rules)
        event.min_normalized_distance = min(event.min_normalized_distance, a.normalized_distance)

        # Promote to ACTIVE if persistence threshold satisfied
        if event.state == EventState.CANDIDATE:
            if (timestamp - event.start_time) >= self.min_persistence:
                event.state = EventState.ACTIVE
                logger.warning(
                    f"Alert Activated: {event.event_id} (Ped #{pair[0]}, Veh #{pair[1]}) | "
                    f"Level: {event.highest_risk_level.value.upper()} | Rules: {list(event.triggered_rules)}"
                )

        return event

    def _resolve_event(self, pair: Tuple[int, int], timestamp: float) -> Optional[EventRecord]:
        """Transition event to RESOLVED, enter cooldown, and archive if it was ACTIVE."""
        event = self.active_events.pop(pair, None)
        if not event:
            return None

        event.end_time = timestamp
        event.state = EventState.RESOLVED

        # Enter cooldown
        self.cooldown_tracker[pair] = timestamp

        if event.highest_score >= 2 and (event.end_time - event.start_time) >= self.min_persistence:
            self.completed_events.append(event)
            logger.info(f"Event Resolved: {event.event_id} | Duration: {event.end_time - event.start_time:.2f}s")
            return event
        return None
